/*
 * voxtral.c - Main API for Voxtral Realtime 4B inference
 *
 * Orchestrates the full pipeline:
 *   Load weights -> WAV -> Mel -> Encoder -> Adapter -> Decoder -> Tokenizer -> Text
 */

#include "voxtral.h"
#include "voxtral_kernels.h"
#include "voxtral_safetensors.h"
#include "voxtral_audio.h"
#include "voxtral_tokenizer.h"
#ifdef USE_CUDA
#include "voxtral_cuda.h"
#endif
#ifdef USE_METAL
#include "voxtral_metal.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <sys/time.h>

/* Global verbose flag */
int vox_verbose = 0;

/* Streaming output: when non-NULL, tokens are decoded and written here ASAP */
FILE *vox_stream_output = NULL;

/* ========================================================================
 * Decoder timing conditioning (t_cond + per-layer ada_scale)
 * ======================================================================== */

static void vox_compute_time_embedding(float *out, float t_value) {
    /* Matches vLLM MistralDecoderLayer time embedding:
     * inv_freq = exp(-log(10000) * arange(dim/2) / (dim/2))
     * emb = t * inv_freq
     * out = [cos(emb), sin(emb)] */
    int dim = VOX_DEC_DIM;
    int half = dim / 2;
    float log_theta = logf(10000.0f);
    for (int i = 0; i < half; i++) {
        float inv_freq = expf(-log_theta * (float)i / (float)half);
        float emb = t_value * inv_freq;
        out[i] = cosf(emb);
        out[i + half] = sinf(emb);
    }
}

static void vox_update_time_conditioning(vox_ctx_t *ctx) {
    if (!ctx) return;
    vox_compute_time_embedding(ctx->t_cond, (float)ctx->delay_tokens);

    size_t n = (size_t)VOX_DEC_LAYERS * VOX_DEC_DIM;
    if (!ctx->ada_scale) {
        ctx->ada_scale = (float *)malloc(n * sizeof(float));
    }
    if (!ctx->ada_scale) return;

    /* Precompute per-layer ada_scale = ada_up(GELU(ada_down(t_cond))). */
    float hidden[VOX_ADA_NORM_DIM];
    for (int layer = 0; layer < VOX_DEC_LAYERS; layer++) {
        const vox_dec_layer_t *l = &ctx->decoder.layers[layer];

        /* hidden = ada_down @ t_cond  (32 x 3072) */
        for (int i = 0; i < VOX_ADA_NORM_DIM; i++) {
            float sum = 0.0f;
            const float *row = l->ada_norm_down + (size_t)i * VOX_DEC_DIM;
            for (int j = 0; j < VOX_DEC_DIM; j++) sum += row[j] * ctx->t_cond[j];
            hidden[i] = sum;
        }
        vox_gelu(hidden, VOX_ADA_NORM_DIM);

        float *scale = ctx->ada_scale + (size_t)layer * VOX_DEC_DIM;
        /* scale = ada_up @ hidden  (3072 x 32) */
        for (int i = 0; i < VOX_DEC_DIM; i++) {
            float sum = 0.0f;
            const float *row = l->ada_norm_up + (size_t)i * VOX_ADA_NORM_DIM;
            for (int j = 0; j < VOX_ADA_NORM_DIM; j++) sum += row[j] * hidden[j];
            scale[i] = sum;
        }
    }
}

/* ========================================================================
 * Internal load functions (defined in encoder/decoder .c files)
 * ======================================================================== */

extern int vox_encoder_load(vox_encoder_t *enc, safetensors_file_t *sf);
extern int vox_decoder_load(vox_decoder_t *dec, safetensors_file_t *sf);

/* ========================================================================
 * Adapter Weight Loading
 * ======================================================================== */

static uint16_t *load_bf16_direct(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "adapter: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_bf16_direct(sf, t);
}

static int vox_adapter_load(vox_adapter_t *adapter, safetensors_file_t *sf) {
    adapter->linear0_weight_bf16 = load_bf16_direct(sf,
        "mm_streams_embeddings.embedding_module.audio_language_projection.0.weight");
    adapter->linear1_weight_bf16 = load_bf16_direct(sf,
        "mm_streams_embeddings.embedding_module.audio_language_projection.2.weight");

    if (!adapter->linear0_weight_bf16 || !adapter->linear1_weight_bf16) return -1;
    return 0;
}

/* ========================================================================
 * Model Loading
 * ======================================================================== */

vox_ctx_t *vox_load(const char *model_dir) {
    vox_ctx_t *ctx = (vox_ctx_t *)calloc(1, sizeof(vox_ctx_t));
    if (!ctx) return NULL;

    strncpy(ctx->model_dir, model_dir, sizeof(ctx->model_dir) - 1);
    ctx->delay_tokens = 6; /* 480ms default */
    ctx->use_bf16 = 1; /* bf16 mmap direct mode */

    /* Open safetensors file */
    char path[1024];
    snprintf(path, sizeof(path), "%s/consolidated.safetensors", model_dir);

    if (vox_verbose >= 2)
        fprintf(stderr, "Loading model from %s\n", path);

    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) {
        fprintf(stderr, "vox_load: cannot open %s\n", path);
        free(ctx);
        return NULL;
    }
    ctx->safetensors = sf;

    if (vox_verbose >= 1)
        fprintf(stderr, "Loading weights...\n");
    if (vox_encoder_load(&ctx->encoder, sf) != 0) {
        fprintf(stderr, "vox_load: failed to load encoder\n");
        vox_free(ctx);
        return NULL;
    }

    if (vox_adapter_load(&ctx->adapter, sf) != 0) {
        fprintf(stderr, "vox_load: failed to load adapter\n");
        vox_free(ctx);
        return NULL;
    }

    if (vox_decoder_load(&ctx->decoder, sf) != 0) {
        fprintf(stderr, "vox_load: failed to load decoder\n");
        vox_free(ctx);
        return NULL;
    }

    /* Precompute time conditioning for the decoder (t_cond + per-layer ada_scale). */
    vox_update_time_conditioning(ctx);

#ifdef USE_METAL
    /* Pre-warm Metal bf16->f16 weight cache to avoid first-token spike */
    if (vox_metal_available()) {
        if (vox_verbose >= 2)
            fprintf(stderr, "Pre-warming Metal weight cache...\n");

        /* Encoder weights (32 layers) */
        for (int i = 0; i < VOX_ENC_LAYERS; i++) {
            vox_enc_layer_t *l = &ctx->encoder.layers[i];
            size_t enc_attn = (size_t)(VOX_ENC_HEADS * VOX_ENC_HEAD_DIM) * VOX_ENC_DIM;
            size_t enc_wo   = (size_t)VOX_ENC_DIM * (VOX_ENC_HEADS * VOX_ENC_HEAD_DIM);
            size_t enc_ffn1 = (size_t)VOX_ENC_HIDDEN * VOX_ENC_DIM;
            size_t enc_ffn2 = (size_t)VOX_ENC_DIM * VOX_ENC_HIDDEN;
            vox_metal_warmup_bf16(l->wq_weight_bf16, enc_attn);
            vox_metal_warmup_bf16(l->wk_weight_bf16, enc_attn);
            vox_metal_warmup_bf16(l->wv_weight_bf16, enc_attn);
            vox_metal_warmup_bf16(l->wo_weight_bf16, enc_wo);
            vox_metal_warmup_bf16(l->w1_weight_bf16, enc_ffn1);
            vox_metal_warmup_bf16(l->w2_weight_bf16, enc_ffn2);
            vox_metal_warmup_bf16(l->w3_weight_bf16, enc_ffn1);
        }

        /* Adapter weights */
        vox_metal_warmup_bf16(ctx->adapter.linear0_weight_bf16,
                              (size_t)VOX_DEC_DIM * (VOX_ENC_DIM * VOX_DOWNSAMPLE));
        vox_metal_warmup_bf16(ctx->adapter.linear1_weight_bf16,
                              (size_t)VOX_DEC_DIM * VOX_DEC_DIM);

        /* Decoder weights (26 layers) */
        for (int i = 0; i < VOX_DEC_LAYERS; i++) {
            vox_dec_layer_t *l = &ctx->decoder.layers[i];
            size_t dec_q  = (size_t)(VOX_DEC_HEADS * VOX_DEC_HEAD_DIM) * VOX_DEC_DIM;
            size_t dec_kv = (size_t)(VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM) * VOX_DEC_DIM;
            size_t dec_wo = (size_t)VOX_DEC_DIM * (VOX_DEC_HEADS * VOX_DEC_HEAD_DIM);
            size_t dec_f1 = (size_t)VOX_DEC_HIDDEN * VOX_DEC_DIM;
            size_t dec_f2 = (size_t)VOX_DEC_DIM * VOX_DEC_HIDDEN;
            vox_metal_warmup_bf16(l->wq_weight_bf16, dec_q);
            vox_metal_warmup_bf16(l->wk_weight_bf16, dec_kv);
            vox_metal_warmup_bf16(l->wv_weight_bf16, dec_kv);
            vox_metal_warmup_bf16(l->wo_weight_bf16, dec_wo);
            vox_metal_warmup_bf16(l->w1_weight_bf16, dec_f1);
            vox_metal_warmup_bf16(l->w2_weight_bf16, dec_f2);
            vox_metal_warmup_bf16(l->w3_weight_bf16, dec_f1);
        }

        /* Token embeddings (also used as logits projection) */
        vox_metal_warmup_bf16(ctx->decoder.tok_embeddings_bf16,
                              (size_t)VOX_VOCAB_SIZE * VOX_DEC_DIM);

        if (vox_verbose >= 1)
            fprintf(stderr, "Metal GPU: %.1f MB\n",
                    vox_metal_memory_used() / (1024.0 * 1024.0));
    }
#endif

    if (vox_verbose >= 1)
        fprintf(stderr, "Model loaded.\n");
    return ctx;
}

/* ========================================================================
 * Model Free
 * ======================================================================== */

void vox_free(vox_ctx_t *ctx) {
    if (!ctx) return;

    /* Free f32 weights that were converted/copied from safetensors. */
    #define FREE0(p) do { free(p); (p) = NULL; } while (0)

    /* Encoder conv stem */
    FREE0(ctx->encoder.conv0_weight);
    FREE0(ctx->encoder.conv0_bias);
    FREE0(ctx->encoder.conv1_weight);
    FREE0(ctx->encoder.conv1_bias);

    for (int i = 0; i < VOX_ENC_LAYERS; i++) {
        vox_enc_layer_t *l = &ctx->encoder.layers[i];
        FREE0(l->wq_bias);
        FREE0(l->wv_bias);
        FREE0(l->wo_bias);
        FREE0(l->attention_norm);
        FREE0(l->w2_bias);
        FREE0(l->ffn_norm);
    }
    FREE0(ctx->encoder.norm);

    /* Decoder */
    for (int i = 0; i < VOX_DEC_LAYERS; i++) {
        vox_dec_layer_t *l = &ctx->decoder.layers[i];
        FREE0(l->ada_norm_down);
        FREE0(l->ada_norm_up);
        FREE0(l->attention_norm);
        FREE0(l->ffn_norm);
    }
    FREE0(ctx->decoder.norm);

    #undef FREE0

    free(ctx->kv_cache_k);
    free(ctx->kv_cache_v);
    free(ctx->ada_scale);

    /* Persistent decoder buffers */
    free(ctx->dec_x);
    free(ctx->dec_x_norm);
    free(ctx->dec_q);
    free(ctx->dec_k);
    free(ctx->dec_v);
    free(ctx->dec_attn_out);
    free(ctx->dec_proj_out);
    free(ctx->dec_gate);
    free(ctx->dec_up);
    free(ctx->dec_ffn_out);
    free(ctx->dec_rope_freqs);
    free(ctx->dec_logits);

    if (ctx->safetensors) {
        safetensors_close((safetensors_file_t *)ctx->safetensors);
    }

    free(ctx);
}

/* ========================================================================
 * Transcription Constants
 * ======================================================================== */

/*
 * Special token IDs (Tekken):
 *   BOS = 1, EOS = 2, [STREAMING_PAD] = 32
 *
 * Decoder input: inputs_embeds[pos] = adapter_out[pos] + tok_embed(input_id)
 * Prompt: [BOS] + [STREAMING_PAD] * (32 + delay_tokens)
 */
#define TOKEN_BOS          1
#define TOKEN_EOS          2
#define TOKEN_STREAMING_PAD 32

#define RAW_AUDIO_LENGTH_PER_TOK 1280
#define OFFLINE_STREAMING_BUFFER_TOKENS 10

/* Overlap in mel frames between encoder chunks (>= 750*2 for sliding window) */
#define OVERLAP_MEL    1504
/* New mel frames per chunk for streaming (~12.5 sec) */
#define STREAM_CHUNK_NEW_MEL  2000
/* First chunk minimum mel frames (enough for 39 prompt adapter tokens) */
#define STREAM_FIRST_CHUNK_MIN_MEL  312

static void trim_ascii_whitespace(char *s) {
    if (!s) return;
    size_t len = strlen(s);
    size_t start = 0;
    while (start < len && isspace((unsigned char)s[start])) start++;
    size_t end = len;
    while (end > start && isspace((unsigned char)s[end - 1])) end--;
    if (start > 0) memmove(s, s + start, end - start);
    s[end - start] = '\0';
}

/* Convert a single token embedding from bf16 to f32 */
static void tok_embed_bf16_to_f32(float *dst, const uint16_t *tok_emb_bf16,
                                  int token_id, int dim) {
    const uint16_t *src = tok_emb_bf16 + (size_t)token_id * dim;
    for (int i = 0; i < dim; i++) {
        uint32_t f32_bits = ((uint32_t)src[i]) << 16;
        memcpy(&dst[i], &f32_bits, sizeof(float));
    }
}

/* ========================================================================
 * Streaming API Implementation
 * ======================================================================== */

struct vox_stream {
    vox_ctx_t *ctx;
    vox_tokenizer_t *tokenizer;

    /* Incremental mel */
    vox_mel_ctx_t *mel_ctx;
    int real_samples_fed;

    /* Encoder chunk tracking */
    int mel_cursor;
    int chunk_num;
    int overlap_mel;
    int chunk_new_mel;
    int first_chunk_min_mel;
    int chunk_user_override;

    /* Adapter output buffer (growing) */
    float *adapter_buf;
    int total_adapter;
    int adapter_cap;

    /* Decoder state */
    int decoder_started;
    int gen_pos;        /* next adapter position for generation */
    int prev_token;
    int eos_seen;
    int finished;       /* vox_stream_finish() called */

    /* Pending token queue (circular buffer of const char* into tokenizer) */
    const char **token_queue;
    int queue_head;     /* next to read */
    int queue_tail;     /* next to write */
    int queue_cap;

    /* Decoder working buffers */
    float *logits;
    float *step_embed;
    float *tok_tmp;

    /* Processing interval: minimum samples between encoder/decoder runs */
    int interval_samples;       /* 0 = process on every feed() */
    int samples_since_process;

    /* Timing */
    double encoder_ms;
    double decoder_ms;
    int n_generated;
};

static void stream_enqueue_token(vox_stream_t *s, const char *piece) {
    /* Grow queue if full */
    int next_tail = (s->queue_tail + 1) % s->queue_cap;
    if (next_tail == s->queue_head) {
        int old_cap = s->queue_cap;
        int new_cap = old_cap * 2;
        const char **new_q = (const char **)malloc((size_t)new_cap * sizeof(const char *));
        if (!new_q) return;
        /* Copy old entries in order */
        int n = 0;
        for (int i = s->queue_head; i != s->queue_tail; i = (i + 1) % old_cap) {
            new_q[n++] = s->token_queue[i];
        }
        free(s->token_queue);
        s->token_queue = new_q;
        s->queue_head = 0;
        s->queue_tail = n;
        s->queue_cap = new_cap;
        next_tail = (s->queue_tail + 1) % s->queue_cap;
    }
    s->token_queue[s->queue_tail] = piece;
    s->queue_tail = next_tail;
    if (vox_stream_output) {
        fputs(piece, vox_stream_output);
        fflush(vox_stream_output);
    }
}

/* Run encoder chunks on available mel, append adapter tokens */
static void stream_run_encoder(vox_stream_t *s) {
    int total_mel = 0;
    float *mel_data = vox_mel_data(s->mel_ctx, &total_mel);
    int dim = VOX_DEC_DIM;

    int new_mel = total_mel - s->mel_cursor;
    int need_mel = (s->chunk_num == 0) ? s->first_chunk_min_mel : s->chunk_new_mel;

    while (new_mel >= need_mel || (s->finished && new_mel > 0)) {
        int overlap = (s->chunk_num == 0) ? 0 : s->overlap_mel;
        int slice_start = s->mel_cursor - overlap;
        int actual_overlap_mel;
        if (slice_start < 0) {
            /* Overlap reaches back to beginning — discard previous partial
             * results and re-encode from scratch as a single pass. */
            slice_start = 0;
            actual_overlap_mel = 0;
            s->total_adapter = 0;
        } else {
            actual_overlap_mel = s->mel_cursor - slice_start;
        }
        int slice_end = s->mel_cursor + new_mel;
        if (!s->finished && new_mel > s->chunk_new_mel)
            slice_end = s->mel_cursor + s->chunk_new_mel;
        if (slice_end > total_mel) slice_end = total_mel;
        int slice_len = slice_end - slice_start;

        struct timeval t0, t1;
        gettimeofday(&t0, NULL);

        float *adapter_chunk = NULL;
        int chunk_tokens = 0;
        int used_cuda = 0;

#ifdef USE_CUDA
        if (vox_cuda_encode_adapter(&adapter_chunk, &chunk_tokens,
                                    s->ctx,
                                    mel_data + (size_t)slice_start * VOX_MEL_BINS,
                                    slice_len,
                                    actual_overlap_mel)) {
            used_cuda = 1;
        }
#endif

        if (!used_cuda) {
            /* Fallback: CPU encoder + adapter (matmuls may still be CUDA). */
            int enc_len = 0;
            float *enc_out = vox_encoder_forward(s->ctx,
                mel_data + (size_t)slice_start * VOX_MEL_BINS,
                slice_len, &enc_len);
            if (!enc_out) break;

            int overlap_enc = actual_overlap_mel / 2;
            int new_enc_len = enc_len - overlap_enc;
            new_enc_len = (new_enc_len / 4) * 4;

            if (new_enc_len > 0) {
                adapter_chunk = vox_adapter_forward(s->ctx,
                    enc_out + (size_t)overlap_enc * VOX_ENC_DIM,
                    new_enc_len, &chunk_tokens);
                free(enc_out);
                if (!adapter_chunk) break;
            } else {
                free(enc_out);
                adapter_chunk = NULL;
                chunk_tokens = 0;
            }
        }

        gettimeofday(&t1, NULL);
        s->encoder_ms += (t1.tv_sec - t0.tv_sec) * 1000.0 +
                         (t1.tv_usec - t0.tv_usec) / 1000.0;

        if (chunk_tokens > 0 && adapter_chunk) {
            if (s->total_adapter + chunk_tokens > s->adapter_cap) {
                int new_cap = s->adapter_cap ? s->adapter_cap * 2 : 256;
                while (new_cap < s->total_adapter + chunk_tokens) new_cap *= 2;
                float *tmp = (float *)realloc(s->adapter_buf,
                    (size_t)new_cap * dim * sizeof(float));
                if (!tmp) { free(adapter_chunk); break; }
                s->adapter_buf = tmp;
                s->adapter_cap = new_cap;
            }
            memcpy(s->adapter_buf + (size_t)s->total_adapter * dim,
                   adapter_chunk, (size_t)chunk_tokens * dim * sizeof(float));
            free(adapter_chunk);
            s->total_adapter += chunk_tokens;
        } else {
            free(adapter_chunk);
        }

        s->mel_cursor = slice_end;
        s->chunk_num++;

        new_mel = total_mel - s->mel_cursor;
        need_mel = s->chunk_new_mel;
    }
}

/* Run decoder: prefill if needed, then generate tokens while adapter available */
static void stream_run_decoder(vox_stream_t *s) {
    struct timeval t0, t1;
    int dim = VOX_DEC_DIM;
    int prompt_len = 1 + 32 + s->ctx->delay_tokens;
    uint16_t *tok_emb_bf16 = s->ctx->decoder.tok_embeddings_bf16;
    const int eos_is_terminal = s->finished; /* EOS is only final once input is finished (offline right-pad added). */

    /* Prefill when we have enough adapter tokens */
    if (!s->decoder_started && s->total_adapter >= prompt_len) {
        gettimeofday(&t0, NULL);

        float *prompt_embeds = (float *)malloc((size_t)prompt_len * dim * sizeof(float));
        if (!prompt_embeds) return;

        for (int i = 0; i < prompt_len; i++) {
            int tok = (i == 0) ? TOKEN_BOS : TOKEN_STREAMING_PAD;
            tok_embed_bf16_to_f32(s->tok_tmp, tok_emb_bf16, tok, dim);
            const float *a = s->adapter_buf + (size_t)i * dim;
            float *dst = prompt_embeds + (size_t)i * dim;
            for (int j = 0; j < dim; j++) dst[j] = a[j] + s->tok_tmp[j];
        }

        s->ctx->kv_cache_len = 0;
        s->ctx->kv_pos_offset = 0;
#ifdef USE_CUDA
        vox_cuda_kv_cache_reset();
#endif
        free(s->ctx->kv_cache_k); s->ctx->kv_cache_k = NULL;
        free(s->ctx->kv_cache_v); s->ctx->kv_cache_v = NULL;

        int prefill_count = prompt_len - 1;
        vox_decoder_prefill(s->ctx, prompt_embeds, prefill_count);

        memcpy(s->step_embed, prompt_embeds + (size_t)prefill_count * dim,
               (size_t)dim * sizeof(float));
        free(prompt_embeds);

        /* Pass logits=NULL to avoid copying full vocab logits back to host on CUDA.
         * The CPU fallback path will use ctx->dec_logits scratch space. */
        s->prev_token = vox_decoder_forward(s->ctx, s->step_embed, NULL);
        s->n_generated++;

        /* Enqueue if it's a text token */
        if (s->prev_token != TOKEN_EOS && s->prev_token >= 1000) {
            const char *piece = vox_tokenizer_decode(s->tokenizer, s->prev_token);
            if (piece) stream_enqueue_token(s, piece);
        }
        if (s->prev_token == TOKEN_EOS) {
            if (eos_is_terminal) {
                s->eos_seen = 1;
            } else {
                /* In incremental/online mode we may see EOS at chunk boundaries.
                 * Keep going by treating it as padding rather than final stop. */
                s->prev_token = TOKEN_STREAMING_PAD;
            }
        }

        s->gen_pos = prompt_len;
        s->decoder_started = 1;

        gettimeofday(&t1, NULL);
        s->decoder_ms += (t1.tv_sec - t0.tv_sec) * 1000.0 +
                         (t1.tv_usec - t0.tv_usec) / 1000.0;

        if (vox_verbose >= 2)
            fprintf(stderr, "Decoder started (prefill %d, first token: %d)\n",
                    prefill_count, s->prev_token);
    }

    /* Generate tokens while adapter tokens are available */
    if (s->decoder_started && !s->eos_seen) {
        gettimeofday(&t0, NULL);
        int gen_before = s->n_generated;
        while (s->gen_pos < s->total_adapter) {
            tok_embed_bf16_to_f32(s->tok_tmp, tok_emb_bf16, s->prev_token, dim);
            const float *a = s->adapter_buf + (size_t)s->gen_pos * dim;
            for (int j = 0; j < dim; j++)
                s->step_embed[j] = a[j] + s->tok_tmp[j];

            s->prev_token = vox_decoder_forward(s->ctx, s->step_embed, NULL);
            s->n_generated++;

            if (s->prev_token != TOKEN_EOS && s->prev_token >= 1000) {
                const char *piece = vox_tokenizer_decode(s->tokenizer, s->prev_token);
                if (piece) stream_enqueue_token(s, piece);
            }

            s->gen_pos++;
            if (s->prev_token == TOKEN_EOS) {
                if (eos_is_terminal) {
                    s->eos_seen = 1;
                    break;
                }
                /* Provisional EOS: continue decoding using padding token. */
                s->prev_token = TOKEN_STREAMING_PAD;
            }
        }
        if (s->n_generated > gen_before) {
            gettimeofday(&t1, NULL);
            s->decoder_ms += (t1.tv_sec - t0.tv_sec) * 1000.0 +
                             (t1.tv_usec - t0.tv_usec) / 1000.0;
        }
    }
}

vox_stream_t *vox_stream_init(vox_ctx_t *ctx) {
    vox_stream_t *s = (vox_stream_t *)calloc(1, sizeof(vox_stream_t));
    if (!s) return NULL;

    s->ctx = ctx;

    /* Load tokenizer */
    char tok_path[1024];
    snprintf(tok_path, sizeof(tok_path), "%s/tekken.json", ctx->model_dir);
    s->tokenizer = vox_tokenizer_load(tok_path);
    if (!s->tokenizer) { free(s); return NULL; }

    /* Initialize incremental mel with 32 left-pad tokens of silence */
    s->mel_ctx = vox_mel_ctx_init(32 * RAW_AUDIO_LENGTH_PER_TOK);
    if (!s->mel_ctx) {
        vox_tokenizer_free(s->tokenizer);
        free(s);
        return NULL;
    }

    /* Token queue */
    s->queue_cap = 256;
    s->token_queue = (const char **)malloc((size_t)s->queue_cap * sizeof(const char *));

    /* Decoder working buffers */
    int dim = VOX_DEC_DIM;
    s->logits = (float *)malloc(VOX_VOCAB_SIZE * sizeof(float));
    s->step_embed = (float *)malloc(dim * sizeof(float));
    s->tok_tmp = (float *)malloc(dim * sizeof(float));

    if (!s->token_queue || !s->logits || !s->step_embed || !s->tok_tmp) {
        vox_stream_free(s);
        return NULL;
    }

    /* Encoder chunking defaults */
    s->overlap_mel = OVERLAP_MEL;
    s->chunk_new_mel = STREAM_CHUNK_NEW_MEL;
    s->first_chunk_min_mel = STREAM_FIRST_CHUNK_MIN_MEL;

    /* Optional offline tuning: increase chunk size to reduce overlap re-encode overhead.
     * This trades latency for throughput (useful for long files). */
    {
        const char *v = getenv("VOX_STREAM_CHUNK_NEW_MEL");
        if (v && v[0]) {
            int n = atoi(v);
            if (n >= 256 && n <= 32768) {
                s->chunk_new_mel = n;
                s->chunk_user_override = 1;
            }
        }
    }

    return s;
}

int vox_stream_feed(vox_stream_t *s, const float *samples, int n_samples) {
    if (!s || s->finished || n_samples <= 0) return -1;

    vox_mel_feed(s->mel_ctx, samples, n_samples);
    s->real_samples_fed += n_samples;
    s->samples_since_process += n_samples;

    if (s->interval_samples == 0 ||
        s->samples_since_process >= s->interval_samples) {
        /* Heuristic: if we are fed a large chunk (offline file or big pipe) and
         * the user didn't explicitly choose a chunk size, increase chunk_new_mel
         * so the encoder doesn't re-encode huge overlaps many times. */
        if (!s->chunk_user_override && s->chunk_num == 0 &&
            n_samples >= VOX_SAMPLE_RATE * 20) {
            /* For long offline inputs, increase chunk size to reduce overlap re-encode
             * overhead, but keep it bounded so encoder attention doesn't explode in
             * time/memory (CUDA path materializes scores). */
            s->chunk_new_mel = 4096;
        }
        stream_run_encoder(s);
        stream_run_decoder(s);
        s->samples_since_process = 0;
    }
    return 0;
}

int vox_stream_finish(vox_stream_t *s) {
    if (!s || s->finished) return -1;
    s->finished = 1;

    int n_delay_tokens = s->ctx->delay_tokens;

    /* Right padding: align + buffer tokens */
    int align_pad = (RAW_AUDIO_LENGTH_PER_TOK -
        (s->real_samples_fed % RAW_AUDIO_LENGTH_PER_TOK)) % RAW_AUDIO_LENGTH_PER_TOK;
    int n_right_pad_tokens = (n_delay_tokens + 1) + OFFLINE_STREAMING_BUFFER_TOKENS;
    int right_pad = align_pad + n_right_pad_tokens * RAW_AUDIO_LENGTH_PER_TOK;

    /* Feed right padding zeros */
    float zero_buf[4096];
    memset(zero_buf, 0, sizeof(zero_buf));
    int remaining = right_pad;
    while (remaining > 0) {
        int chunk = remaining > 4096 ? 4096 : remaining;
        vox_mel_feed(s->mel_ctx, zero_buf, chunk);
        remaining -= chunk;
    }

    /* Finalize mel */
    vox_mel_finish(s->mel_ctx, 0);

    if (vox_verbose >= 2)
        fprintf(stderr, "Stream finished: %d real samples (%.1f sec)\n",
                s->real_samples_fed, (float)s->real_samples_fed / VOX_SAMPLE_RATE);

    /* Process remaining encoder chunks and generate remaining tokens */
    stream_run_encoder(s);
    stream_run_decoder(s);

    const char *force_timing_env = getenv("VOX_PRINT_TIMINGS");
    int force_timing = (force_timing_env && force_timing_env[0] && force_timing_env[0] != '0');

    if (vox_verbose >= 1 || force_timing) {
        /* Ensure stdout transcription output ends with newline before timing */
        if (vox_stream_output) {
            fputs("\n", vox_stream_output);
            fflush(vox_stream_output);
        }
        fprintf(stderr, "Encoder: %d mel -> %d tokens (%.0f ms)\n",
                s->mel_cursor, s->total_adapter, s->encoder_ms);
        if (s->n_generated > 0)
            fprintf(stderr, "Decoder: %d tokens in %.0f ms (%.1f ms/token)\n",
                    s->n_generated, s->decoder_ms, s->decoder_ms / s->n_generated);
    }
    return 0;
}

int vox_stream_get(vox_stream_t *s, const char **out_tokens, int max) {
    if (!s || max <= 0) return 0;
    int count = 0;
    while (count < max && s->queue_head != s->queue_tail) {
        out_tokens[count++] = s->token_queue[s->queue_head];
        s->queue_head = (s->queue_head + 1) % s->queue_cap;
    }
    return count;
}

void vox_stream_free(vox_stream_t *s) {
    if (!s) return;
    vox_mel_free(s->mel_ctx);
    if (s->tokenizer) vox_tokenizer_free(s->tokenizer);
    free(s->adapter_buf);
    free(s->token_queue);
    free(s->logits);
    free(s->step_embed);
    free(s->tok_tmp);
    free(s);
}

/* ========================================================================
 * Convenience Functions (built on streaming API)
 * ======================================================================== */

char *vox_transcribe_audio(vox_ctx_t *ctx, const float *samples, int n_samples) {
    vox_stream_t *s = vox_stream_init(ctx);
    if (!s) return NULL;

    vox_stream_feed(s, samples, n_samples);
    vox_stream_finish(s);

    /* Collect all tokens into a string */
    size_t text_cap = 1024;
    size_t text_len = 0;
    char *text = (char *)malloc(text_cap);
    text[0] = '\0';

    const char *tokens[64];
    int n;
    while ((n = vox_stream_get(s, tokens, 64)) > 0) {
        for (int i = 0; i < n; i++) {
            size_t piece_len = strlen(tokens[i]);
            if (text_len + piece_len + 1 > text_cap) {
                while (text_len + piece_len + 1 > text_cap) text_cap *= 2;
                text = (char *)realloc(text, text_cap);
            }
            memcpy(text + text_len, tokens[i], piece_len);
            text_len += piece_len;
            text[text_len] = '\0';
        }
    }

    vox_stream_free(s);
    trim_ascii_whitespace(text);
    return text;
}

char *vox_transcribe_stdin(vox_ctx_t *ctx) {
    /* Read first 4 bytes to detect format */
    uint8_t header[4];
    size_t hdr_read = fread(header, 1, 4, stdin);
    if (hdr_read < 4) {
        fprintf(stderr, "vox_transcribe_stdin: not enough data on stdin\n");
        return NULL;
    }

    /* WAV detected: buffer all, parse, transcribe as offline */
    if (memcmp(header, "RIFF", 4) == 0) {
        if (vox_verbose >= 2)
            fprintf(stderr, "Detected WAV format on stdin\n");

        size_t capacity = 1024 * 1024;
        size_t size = 4;
        uint8_t *buf = (uint8_t *)malloc(capacity);
        if (!buf) return NULL;
        memcpy(buf, header, 4);
        while (1) {
            if (size == capacity) {
                capacity *= 2;
                uint8_t *tmp = (uint8_t *)realloc(buf, capacity);
                if (!tmp) { free(buf); return NULL; }
                buf = tmp;
            }
            size_t n = fread(buf + size, 1, capacity - size, stdin);
            if (n == 0) break;
            size += n;
        }
        if (vox_verbose >= 2)
            fprintf(stderr, "Read %zu bytes from stdin\n", size);

        if (size < 44 || memcmp(buf + 8, "WAVE", 4) != 0) {
            fprintf(stderr, "Invalid WAV data on stdin\n");
            free(buf);
            return NULL;
        }

        int channels = 0, sample_rate = 0, bits_per_sample = 0, audio_format = 0;
        const uint8_t *pcm_data = NULL;
        int pcm_size = 0;
        const uint8_t *p = buf + 12;
        const uint8_t *end = buf + size;
        while (p + 8 <= end) {
            uint32_t chunk_size = (uint32_t)p[4] | ((uint32_t)p[5]<<8) |
                                  ((uint32_t)p[6]<<16) | ((uint32_t)p[7]<<24);
            if (p + 8 + chunk_size > end) break;
            if (memcmp(p, "fmt ", 4) == 0 && chunk_size >= 16) {
                audio_format = p[8] | (p[9]<<8);
                channels = p[10] | (p[11]<<8);
                sample_rate = (int)((uint32_t)p[12] | ((uint32_t)p[13]<<8) |
                              ((uint32_t)p[14]<<16) | ((uint32_t)p[15]<<24));
                bits_per_sample = p[22] | (p[23]<<8);
            } else if (memcmp(p, "data", 4) == 0) {
                pcm_data = p + 8;
                pcm_size = (int)chunk_size;
                if (pcm_data + pcm_size > end) pcm_size = (int)(end - pcm_data);
            }
            p += 8 + chunk_size;
            if (chunk_size & 1) p++;
        }

        if (audio_format != 1 || bits_per_sample != 16 || !pcm_data || channels < 1) {
            fprintf(stderr, "Unsupported WAV format on stdin\n");
            free(buf);
            return NULL;
        }

        int n_frames = pcm_size / (channels * 2);
        float *samples = (float *)malloc((size_t)n_frames * sizeof(float));
        if (!samples) { free(buf); return NULL; }
        const int16_t *src = (const int16_t *)pcm_data;
        for (int i = 0; i < n_frames; i++) {
            if (channels == 1) {
                samples[i] = src[i] / 32768.0f;
            } else {
                float sum = 0;
                for (int c = 0; c < channels; c++) {
                    int16_t val;
                    memcpy(&val, &src[i * channels + c], sizeof(int16_t));
                    sum += val;
                }
                samples[i] = (sum / channels) / 32768.0f;
            }
        }
        free(buf);

        if (sample_rate != VOX_SAMPLE_RATE) {
            int new_n = (int)((long long)n_frames * VOX_SAMPLE_RATE / sample_rate);
            float *resampled = (float *)malloc((size_t)new_n * sizeof(float));
            if (!resampled) { free(samples); return NULL; }
            for (int i = 0; i < new_n; i++) {
                float src_pos = (float)i * sample_rate / VOX_SAMPLE_RATE;
                int idx = (int)src_pos;
                float frac = src_pos - idx;
                if (idx + 1 < n_frames)
                    resampled[i] = samples[idx] * (1-frac) + samples[idx+1] * frac;
                else
                    resampled[i] = (idx < n_frames) ? samples[idx] : 0.0f;
            }
            free(samples);
            samples = resampled;
            n_frames = new_n;
        }

        if (vox_verbose >= 1)
            fprintf(stderr, "Audio: %d samples (%.1f seconds)\n",
                    n_frames, (float)n_frames / VOX_SAMPLE_RATE);

        /* Use stream API so tokens are emitted incrementally */
        vox_stream_t *s = vox_stream_init(ctx);
        if (!s) { free(samples); return NULL; }
        vox_stream_feed(s, samples, n_frames);
        vox_stream_finish(s);
        free(samples);

        /* Collect and optionally stream tokens */
        size_t text_cap = 1024;
        size_t text_len = 0;
        char *text = (char *)malloc(text_cap);
        text[0] = '\0';

        const char *tokens[64];
        int n;
        while ((n = vox_stream_get(s, tokens, 64)) > 0) {
            for (int i = 0; i < n; i++) {
                size_t piece_len = strlen(tokens[i]);
                if (text_len + piece_len + 1 > text_cap) {
                    while (text_len + piece_len + 1 > text_cap) text_cap *= 2;
                    text = (char *)realloc(text, text_cap);
                }
                memcpy(text + text_len, tokens[i], piece_len);
                text_len += piece_len;
                text[text_len] = '\0';
            }
        }
        vox_stream_free(s);
        trim_ascii_whitespace(text);
        return text;
    }

    /* Raw s16le streaming mode */
    if (vox_verbose >= 2)
        fprintf(stderr, "Streaming raw s16le 16kHz mono from stdin\n");

    vox_stream_t *s = vox_stream_init(ctx);
    if (!s) return NULL;

    /* Optional batching for stdin raw mode to reduce overhead when the input
     * arrives in many small chunks (e.g. ffmpeg pipe). Default is 0 (process on
     * every feed). */
    {
        const char *v = getenv("VOX_STDIN_INTERVAL_SEC");
        if (v && v[0]) {
            float sec = (float)atof(v);
            if (sec < 0.0f) sec = 0.0f;
            vox_set_processing_interval(s, sec);
        }
    }

    /* Feed the 4 peeked header bytes as 2 s16le samples */
    {
        int16_t sv[2];
        memcpy(sv, header, 4);
        float f[2] = { sv[0] / 32768.0f, sv[1] / 32768.0f };
        vox_stream_feed(s, f, 2);
    }

    /* Collect text for non-streaming mode, or emit for streaming */
    size_t text_cap = 1024;
    size_t text_len = 0;
    char *text = (char *)malloc(text_cap);
    text[0] = '\0';

    /* Read loop */
    int16_t raw_buf[4096];
    float fbuf[4096];
    const char *tokens[64];
    int eof_reached = 0;

    while (!eof_reached) {
        size_t nread = fread(raw_buf, sizeof(int16_t), 4096, stdin);
        if (nread == 0) {
            eof_reached = 1;
            vox_stream_finish(s);
        } else {
            for (size_t i = 0; i < nread; i++)
                fbuf[i] = raw_buf[i] / 32768.0f;
            vox_stream_feed(s, fbuf, (int)nread);
        }

        /* Drain pending tokens */
        int n;
        while ((n = vox_stream_get(s, tokens, 64)) > 0) {
            for (int i = 0; i < n; i++) {
                size_t piece_len = strlen(tokens[i]);
                if (text_len + piece_len + 1 > text_cap) {
                    while (text_len + piece_len + 1 > text_cap) text_cap *= 2;
                    text = (char *)realloc(text, text_cap);
                }
                memcpy(text + text_len, tokens[i], piece_len);
                text_len += piece_len;
                text[text_len] = '\0';
            }
        }
    }

    vox_stream_free(s);
    trim_ascii_whitespace(text);
    return text;
}

char *vox_transcribe(vox_ctx_t *ctx, const char *wav_path) {
    int n_samples = 0;
    float *samples = vox_load_wav(wav_path, &n_samples);
    if (!samples) {
        fprintf(stderr, "vox_transcribe: cannot load %s\n", wav_path);
        return NULL;
    }
    if (vox_verbose >= 1)
        fprintf(stderr, "Audio: %d samples (%.1f seconds)\n",
                n_samples, (float)n_samples / VOX_SAMPLE_RATE);

    char *text = vox_transcribe_audio(ctx, samples, n_samples);
    free(samples);
    return text;
}

void vox_set_processing_interval(vox_stream_t *s, float seconds) {
    if (!s) return;
    s->interval_samples = (int)(seconds * VOX_SAMPLE_RATE);
    if (s->interval_samples < 0) s->interval_samples = 0;
}

void vox_set_delay(vox_ctx_t *ctx, int delay_ms) {
    /* Each token represents 80ms (frame_rate=12.5Hz) */
    if (delay_ms < 80) delay_ms = 80;
    if (delay_ms > 2400) delay_ms = 2400;
    ctx->delay_tokens = delay_ms / 80;
    vox_update_time_conditioning(ctx);
}
