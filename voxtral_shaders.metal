/*
 * voxtral_shaders.metal - Metal compute shaders for Voxtral inference
 *
 * GPU kernels for element-wise ops that avoid CPU round-trips when used
 * between MPS matmul calls. All operate on f32 tensors.
 */

#include <metal_stdlib>
using namespace metal;

/* ========================================================================
 * RMSNorm: out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]
 * One threadgroup per row. x: [seq, hidden], weight: [hidden]
 * ======================================================================== */

kernel void rms_norm(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant int &hidden [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];

    device const float *x_row = x + row * hidden;
    device float *out_row = out + row * hidden;

    float local_sum = 0.0f;
    for (int i = tid; i < hidden; i += threads) {
        float val = x_row[i];
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms_inv = rsqrt(shared_sum[0] / float(hidden) + eps);

    for (int i = tid; i < hidden; i += threads) {
        out_row[i] = x_row[i] * rms_inv * weight[i];
    }
}

/* ========================================================================
 * SiLU: x = x / (1 + exp(-x))
 * ======================================================================== */

kernel void silu(
    device float *x [[buffer(0)]],
    constant int &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) {
        float val = x[gid];
        x[gid] = val / (1.0f + exp(-val));
    }
}

/* ========================================================================
 * GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
 * ======================================================================== */

kernel void gelu(
    device float *x [[buffer(0)]],
    constant int &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) {
        float val = x[gid];
        float x3 = val * val * val;
        float inner = 0.7978845608028654f * (val + 0.044715f * x3);
        x[gid] = 0.5f * val * (1.0f + tanh(inner));
    }
}

/* ========================================================================
 * Element-wise ops
 * ======================================================================== */

kernel void add_inplace(
    device float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) a[gid] += b[gid];
}

kernel void mul_inplace(
    device float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) a[gid] *= b[gid];
}

/* x[i] *= (1 + scale[i]) — adaptive RMS norm conditioning */
kernel void ada_scale_mul(
    device float *x [[buffer(0)]],
    device const float *scale [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) x[gid] *= (1.0f + scale[gid]);
}

/* ========================================================================
 * Argmax over a float array. Returns index of max value.
 * One threadgroup, result written to out[0].
 * ======================================================================== */

kernel void argmax_f32(
    device const float *data [[buffer(0)]],
    device int *out [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup float shared_val[256];
    threadgroup int shared_idx[256];

    float best_val = -INFINITY;
    int best_idx = 0;
    for (int i = tid; i < n; i += threads) {
        float v = data[i];
        if (v > best_val) { best_val = v; best_idx = i; }
    }
    shared_val[tid] = best_val;
    shared_idx[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_val[tid + stride] > shared_val[tid]) {
                shared_val[tid] = shared_val[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) out[0] = shared_idx[0];
}

/* ========================================================================
 * Causal masked softmax for attention scores.
 * scores: [n_heads, seq_q, seq_k] (contiguous per head)
 * One threadgroup per (query_position, head) pair.
 *
 * Applies:
 *   - Causal mask: query at q_offset+qi attends to keys 0..q_offset+qi
 *   - Sliding window: keys below max(0, q_pos - window + 1) are masked
 *   - Softmax normalization (numerically stable)
 * ======================================================================== */

kernel void causal_softmax(
    device float *scores [[buffer(0)]],
    constant int &seq_q [[buffer(1)]],
    constant int &seq_k [[buffer(2)]],
    constant int &window_size [[buffer(3)]],
    constant int &q_offset [[buffer(4)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    int qi = group_id % seq_q;
    int head = group_id / seq_q;

    device float *row = scores + ((long)head * seq_q + qi) * seq_k;

    int q_pos = q_offset + qi;
    int valid_end = min(q_pos, seq_k - 1);
    int valid_start = (window_size > 0) ? max(0, q_pos - window_size + 1) : 0;

    threadgroup float shared[256];

    /* Phase 1: apply mask, find row max */
    float local_max = -INFINITY;
    for (int j = tid; j < seq_k; j += tg_size) {
        float val = (j >= valid_start && j <= valid_end) ? row[j] : -INFINITY;
        row[j] = val;
        local_max = fmax(local_max, val);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = fmax(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];

    /* Phase 2: exp(x - max) and sum */
    float local_sum = 0.0f;
    for (int j = tid; j < seq_k; j += tg_size) {
        float val = exp(row[j] - row_max);
        row[j] = val;
        local_sum += val;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / (shared[0] + 1e-10f);

    /* Phase 3: normalize */
    for (int j = tid; j < seq_k; j += tg_size) {
        row[j] *= inv_sum;
    }
}

/* ========================================================================
 * RoPE: apply rotary position embedding in-place.
 * data: [n_heads * head_dim], freqs: [head_dim/2 * 2] = (cos,sin) pairs.
 * One thread per (head, half_dim_index) pair.
 * ======================================================================== */

kernel void rope_apply(
    device float *data [[buffer(0)]],
    device const float *freqs [[buffer(1)]],
    constant int &n_heads [[buffer(2)]],
    constant int &head_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    int half_dim = head_dim / 2;
    int total = n_heads * half_dim;
    if ((int)gid >= total) return;

    int head = (int)gid / half_dim;
    int i = (int)gid % half_dim;

    float cos_val = freqs[i * 2];
    float sin_val = freqs[i * 2 + 1];

    int base = head * head_dim;
    float x0 = data[base + i * 2];
    float x1 = data[base + i * 2 + 1];

    data[base + i * 2]     = x0 * cos_val - x1 * sin_val;
    data[base + i * 2 + 1] = x0 * sin_val + x1 * cos_val;
}

/* ========================================================================
 * KV cache copy: write kv_dim floats to a position in the cache.
 * cache: large buffer, data written at float_offset + gid.
 * ======================================================================== */

kernel void kv_cache_copy(
    device float *cache [[buffer(0)]],
    device const float *data [[buffer(1)]],
    constant int &float_offset [[buffer(2)]],
    constant int &kv_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid < kv_dim) {
        cache[float_offset + gid] = data[gid];
    }
}

/* ========================================================================
 * Single-token decoder attention (seq_q=1).
 * One threadgroup per query head, 128 threads cooperate on dot products.
 * K/V read from the KV cache buffer at a per-layer offset.
 * Uses online softmax (single pass) with SIMD group reductions.
 * 128 threads = 4 SIMD groups of 32. simd_sum for fast dot product.
 * ======================================================================== */

kernel void decoder_attention(
    device const float *Q [[buffer(0)]],
    device const float *K_cache [[buffer(1)]],
    device const float *V_cache [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant int &n_heads [[buffer(4)]],
    constant int &n_kv_heads [[buffer(5)]],
    constant int &head_dim [[buffer(6)]],
    constant int &kv_dim [[buffer(7)]],
    constant int &seq_k [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    constant int &window_size [[buffer(10)]],
    constant int &q_pos [[buffer(11)]],
    uint head_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    if ((int)head_idx >= n_heads) return;

    int gqa_ratio = n_heads / n_kv_heads;
    int kv_head = (int)head_idx / gqa_ratio;

    device const float *q_h = Q + head_idx * head_dim;
    device float *out_h = out + head_idx * head_dim;

    int valid_end = min(q_pos, seq_k - 1);
    int valid_start = (window_size > 0) ? max(0, q_pos - window_size + 1) : 0;

    /* 128 threads = 4 SIMD groups of 32 */
    threadgroup float shared_simd[4];

    /* Each thread loads one Q element (head_dim=128) */
    float q_val = (int)tid < head_dim ? q_h[tid] : 0.0f;

    /* Online softmax: single pass over keys */
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc = 0.0f;

    for (int j = valid_start; j <= valid_end; j++) {
        device const float *k_j = K_cache + j * kv_dim + kv_head * head_dim;

        /* Cooperative dot product using SIMD reductions */
        float partial = (int)tid < head_dim ? q_val * k_j[tid] : 0.0f;
        float simd_partial = simd_sum(partial);

        /* Cross-SIMD reduction: 4 groups → 1 value */
        if (simd_lid == 0) shared_simd[simd_gid] = simd_partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score;
        if (tid == 0) {
            shared_simd[0] = (shared_simd[0] + shared_simd[1] +
                              shared_simd[2] + shared_simd[3]) * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        score = shared_simd[0];

        /* Online softmax update */
        float old_max = running_max;
        running_max = fmax(running_max, score);
        float correction = exp(old_max - running_max);
        running_sum = running_sum * correction + exp(score - running_max);
        acc = acc * correction;

        /* Accumulate weighted V */
        if ((int)tid < head_dim) {
            device const float *v_j = V_cache + j * kv_dim + kv_head * head_dim;
            acc += exp(score - running_max) * v_j[tid];
        }
    }

    /* Normalize and write output */
    if ((int)tid < head_dim) {
        out_h[tid] = acc / (running_sum + 1e-10f);
    }
}
