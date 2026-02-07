#!/usr/bin/env python3
"""
Standalone inference for Voxtral Realtime 4B.
No vLLM or transformers dependency - just PyTorch + safetensors.

Usage:
    pip install torch safetensors soundfile soxr
    python inference.py voxtral-model test_speech.wav

Reconstructed from vLLM source code:
  - vllm/model_executor/models/voxtral_realtime.py (TimeEmbedding, forward)
  - vllm/model_executor/models/mistral.py (MistralDecoderLayer, ada_rms_norm)
  - vllm/model_executor/models/whisper_causal.py (CausalConv1d, CausalEncoder)
  - mistral_common/audio.py (mel_filter_bank)  [re-implemented here]
  - mistral_common/tokens/tokenizers/audio.py (AudioConfig, streaming padding)  [re-implemented here]
"""

import sys, os, json, math, base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
import soundfile as sf

# ============================================================================
# Config (from params.json)
# ============================================================================

# Encoder
ENC_DIM = 1280
ENC_LAYERS = 32
ENC_HEADS = 32
ENC_HEAD_DIM = 64
ENC_HIDDEN = 5120
ENC_KV_HEADS = 32
ENC_WINDOW = 750
ENC_NORM_EPS = 1e-5
ENC_ROPE_THETA = 1_000_000.0

# Use bfloat16 for computation (like vLLM)
USE_BF16 = False

# Decoder
DEC_DIM = 3072
DEC_LAYERS = 26
DEC_HEADS = 32
DEC_HEAD_DIM = 128
DEC_HIDDEN = 9216
DEC_KV_HEADS = 8
DEC_WINDOW = 8192
DEC_NORM_EPS = 1e-5
DEC_ROPE_THETA = 1_000_000.0
VOCAB_SIZE = 131072

# Audio
SAMPLE_RATE = 16000
FRAME_RATE = 12.5
NUM_MEL_BINS = 128
HOP_LENGTH = 160
WINDOW_SIZE = 400
GLOBAL_LOG_MEL_MAX = 1.5
DOWNSAMPLE_FACTOR = 4

# Ada norm
ADA_NORM_DIM = 32

# Streaming
N_LEFT_PAD_TOKENS = 32
TRANSCRIPTION_DELAY_MS = 480

# Special tokens - verified from mistral_common tokenizer
# These are the actual token IDs used by the model (NOT rank + base!)
TOKEN_BOS = 1               # <s>
TOKEN_EOS = 2               # </s>
TOKEN_STREAMING_PAD = 32    # [STREAMING_PAD]
TOKEN_BEGIN_AUDIO = 25      # [BEGIN_AUDIO]
TOKEN_AUDIO = 24            # [AUDIO]

# ============================================================================
# Derived constants
# ============================================================================

RAW_AUDIO_LENGTH_PER_TOK = int(SAMPLE_RATE // FRAME_RATE)  # 1280
AUDIO_LENGTH_PER_TOK = RAW_AUDIO_LENGTH_PER_TOK // HOP_LENGTH  # 8

def num_delay_tokens():
    delay_len = int(TRANSCRIPTION_DELAY_MS / 1000.0 * SAMPLE_RATE)  # 7680
    return num_audio_tokens(delay_len)  # 6

def num_audio_tokens(audio_len):
    if audio_len % HOP_LENGTH != 0:
        audio_len = math.ceil(audio_len / HOP_LENGTH - 1)
    else:
        audio_len = audio_len // HOP_LENGTH
    return math.ceil(audio_len / AUDIO_LENGTH_PER_TOK)

N_DELAY_TOKENS = num_delay_tokens()
N_RIGHT_PAD_TOKENS = (N_DELAY_TOKENS + 1) + 10  # 17

# ============================================================================
# Mel filter bank (Slaney-style, from mistral_common/audio.py)
# ============================================================================

def hertz_to_mel(freq):
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0
    if isinstance(freq, np.ndarray):
        log_region = freq >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep
    return mels

def mel_to_hertz(mels):
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    log_region = mels >= min_log_mel
    freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return freq

def compute_mel_filters():
    num_frequency_bins = 1 + WINDOW_SIZE // 2  # 201
    fft_freqs = np.linspace(0, SAMPLE_RATE // 2, num_frequency_bins)
    mel_min = hertz_to_mel(0.0)
    mel_max = hertz_to_mel(8000.0)
    mel_freqs = np.linspace(mel_min, mel_max, NUM_MEL_BINS + 2)
    filter_freqs = mel_to_hertz(mel_freqs)
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    fb = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))
    enorm = 2.0 / (filter_freqs[2:NUM_MEL_BINS+2] - filter_freqs[:NUM_MEL_BINS])
    fb *= np.expand_dims(enorm, 0)
    return fb  # [201, 128]

# ============================================================================
# Mel spectrogram (from vLLM voxtral.py compute_whisper_melspec)
# ============================================================================

def compute_mel_spectrogram(audio, mel_filters):
    """audio: 1D tensor, mel_filters: [freq_bins, mel_bins] tensor"""
    window = torch.hann_window(WINDOW_SIZE)
    stft = torch.stft(audio, WINDOW_SIZE, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    mel_spec = mel_filters.T @ magnitudes  # [mel_bins, frames]
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    # Use fixed global_log_mel_max from params.json (matching vLLM compute_whisper_melspec
    # which uses config.global_log_mel_max when set, falling back to dynamic max otherwise)
    log_spec = torch.maximum(log_spec, torch.tensor(GLOBAL_LOG_MEL_MAX) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec  # [128, frames]

# ============================================================================
# Audio streaming padding (from mistral_common AudioEncoder.pad)
# ============================================================================

def pad_audio_streaming(audio_array):
    """Pad audio for offline streaming mode (matching mistral_common AudioEncoder.pad).

    Offline streaming: left pad + right pad (align + n_right_pad_tokens).
    Audio is aligned to raw_audio_length_per_tok (1280 samples).
    """
    mult_of = RAW_AUDIO_LENGTH_PER_TOK  # 1280

    # Right pad: align to mult_of + n_right_pad_tokens extra
    n_samples = len(audio_array)
    align_pad = (mult_of - (n_samples % mult_of)) % mult_of
    right_pad = align_pad + N_RIGHT_PAD_TOKENS * mult_of  # 17 * 1280 = 21760

    # Left pad: n_left_pad_tokens * mult_of
    left_pad = N_LEFT_PAD_TOKENS * mult_of  # 32 * 1280 = 40960

    return np.pad(audio_array, (left_pad, right_pad))

# ============================================================================
# Weight loading helpers
# ============================================================================

def get_weight(sf_file, name):
    t = sf_file.get_tensor(name)
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t

def get_weight_optional(sf_file, name):
    try:
        return get_weight(sf_file, name)
    except:
        return None

def permute_qk_weight(w, n_heads, head_dim):
    """Permute wq/wk weights from Mistral format to interleaved RoPE format.

    Mistral consolidated.safetensors stores Q/K weights assuming split-half RoPE.
    This permutes them for interleaved RoPE (is_neox_style=False).

    w: [n_heads * head_dim, hidden_size]
    """
    attn_in = n_heads * head_dim
    attn_out = w.shape[1]
    # Reshape to [n_heads, head_dim/2, 2, hidden_size]
    # Then transpose to [n_heads, 2, head_dim/2, hidden_size]
    # This interleaves: [even_dims, odd_dims] -> [dim0, dim1, dim2, dim3, ...]
    return (
        w.view(n_heads, head_dim // 2, 2, attn_out)
        .transpose(1, 2)
        .reshape(attn_in, attn_out)
    )

def permute_qk_bias(b, n_heads, head_dim):
    """Permute wq/wk bias to match the weight permutation."""
    attn_in = n_heads * head_dim
    return (
        b.view(n_heads, head_dim // 2, 2)
        .transpose(1, 2)
        .reshape(attn_in)
    )

# ============================================================================
# RMSNorm
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-5):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms * self.weight).to(x.dtype)

# ============================================================================
# RoPE
# ============================================================================

def compute_rope_freqs(positions, head_dim, theta):
    """positions: [seq_len] int tensor. Returns cos, sin each [seq_len, head_dim/2]."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = positions.float().unsqueeze(-1) * freqs.unsqueeze(0)  # [seq, hd/2]
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos_f, sin_f, n_heads, head_dim, is_neox_style=False):
    """x: [seq, n_heads*head_dim].

    is_neox_style=False: Interleaved (GPT-J), pairs (0,1), (2,3), ...
    is_neox_style=True: Split halves (NeoX), first half, second half
    """
    seq_len = x.shape[0]
    x = x.view(seq_len, n_heads, head_dim)
    cos_f = cos_f.unsqueeze(1)  # [seq, 1, hd/2]
    sin_f = sin_f.unsqueeze(1)

    if is_neox_style:
        # NeoX style: split into halves
        x1, x2 = x.chunk(2, dim=-1)
        o1 = x1 * cos_f - x2 * sin_f
        o2 = x2 * cos_f + x1 * sin_f
        out = torch.cat([o1, o2], dim=-1)
    else:
        # Interleaved: pairs (0,1), (2,3), (4,5), ...
        x1 = x[..., ::2]   # even indices
        x2 = x[..., 1::2]  # odd indices
        o1 = x1 * cos_f - x2 * sin_f
        o2 = x2 * cos_f + x1 * sin_f
        # Interleave back: stack and flatten
        out = torch.stack([o1, o2], dim=-1).flatten(-2)

    return out.view(seq_len, n_heads * head_dim)

# ============================================================================
# Causal Attention
# ============================================================================

def causal_attention(q, k, v, n_heads, n_kv_heads, head_dim, window,
                     q_start_pos=0, kv_start_pos=0):
    """
    q: [seq_q, n_heads*head_dim]         (queries at absolute positions q_start_pos + i)
    k: [seq_kv, n_kv_heads*head_dim]     (keys   at absolute positions kv_start_pos + j)
    v: [seq_kv, n_kv_heads*head_dim]
    Returns: [seq_q, n_heads*head_dim]
    """
    seq_q = q.shape[0]
    seq_kv = k.shape[0]
    gqa_ratio = n_heads // n_kv_heads
    orig_dtype = q.dtype

    # Reshape to [batch, n_heads, seq, head_dim] for scaled_dot_product_attention
    q = q.view(seq_q, n_heads, head_dim).transpose(0, 1).unsqueeze(0)       # [1, nh, sq, hd]
    k = k.view(seq_kv, n_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)   # [1, nkv, skv, hd]
    v = v.view(seq_kv, n_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)

    # Expand KV heads for GQA
    if gqa_ratio > 1:
        k = k.repeat_interleave(gqa_ratio, dim=1)  # [1, nh, skv, hd]
        v = v.repeat_interleave(gqa_ratio, dim=1)

    # Create causal mask with sliding window
    # qi_abs: [sq, 1], kv_abs: [1, skv]
    qi_abs = (q_start_pos + torch.arange(seq_q)).unsqueeze(1)        # [sq, 1]
    kv_abs = (kv_start_pos + torch.arange(seq_kv)).unsqueeze(0)      # [1, skv]

    # attend if kv_abs <= qi_abs AND kv_abs >= qi_abs - (window-1)
    attn_mask = (kv_abs <= qi_abs) & (kv_abs >= (qi_abs - (window - 1)))

    # Use PyTorch's flash attention (if available) via scaled_dot_product_attention
    out = F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(),
        attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),  # [1, 1, sq, skv]
        scale=1.0 / math.sqrt(head_dim),
        dropout_p=0.0,
    ).to(orig_dtype)

    # Reshape back to [seq_q, n_heads*head_dim]
    return out.squeeze(0).transpose(0, 1).contiguous().view(seq_q, n_heads * head_dim)

# ============================================================================
# Causal Conv1d (from whisper_causal.py)
# ============================================================================

def causal_conv1d(x, weight, bias, stride):
    """x: [1, C_in, L], weight: [C_out, C_in, K], returns [1, C_out, L']"""
    kernel_size = weight.shape[2]
    effective_ks = kernel_size  # dilation=1
    padding_total = effective_ks - stride

    n_frames = (x.shape[-1] - effective_ks + padding_total) / stride + 1
    target_length = (math.ceil(n_frames) - 1) * stride + (effective_ks - padding_total)
    extra_padding = int(target_length - x.shape[-1])

    x = F.pad(x, (padding_total, extra_padding), mode='constant')
    return F.conv1d(x, weight, bias, stride=stride)

# ============================================================================
# TimeEmbedding (from voxtral_realtime.py)
# ============================================================================

def compute_time_embedding(t_value, dim, theta=10000.0):
    """Sinusoidal embedding of scalar t_value into dim-dimensional vector."""
    half_dim = dim // 2
    inv_freq = torch.exp(-math.log(theta) * torch.arange(half_dim).float() / half_dim)
    emb = t_value * inv_freq
    return torch.cat([emb.cos(), emb.sin()])  # [dim]

# ============================================================================
# Encoder forward
# ============================================================================

def encoder_forward(mel, weights, sf_file):
    """mel: [128, frames] -> [seq, 1280]"""
    prefix = "mm_streams_embeddings.embedding_module.whisper_encoder"

    # Conv stem
    mel_3d = mel.unsqueeze(0)  # [1, 128, frames]
    conv0_w = get_weight(sf_file, f"{prefix}.conv_layers.0.conv.weight")
    conv0_b = get_weight(sf_file, f"{prefix}.conv_layers.0.conv.bias")
    conv1_w = get_weight(sf_file, f"{prefix}.conv_layers.1.conv.weight")
    conv1_b = get_weight(sf_file, f"{prefix}.conv_layers.1.conv.bias")

    h = F.gelu(causal_conv1d(mel_3d, conv0_w, conv0_b, stride=1))
    h = F.gelu(causal_conv1d(h, conv1_w, conv1_b, stride=2))
    h = h.squeeze(0).transpose(0, 1)  # [seq, 1280]
    conv_len = h.shape[0]

    # Left-truncate conv output to multiple of DOWNSAMPLE_FACTOR (matching vLLM realtime)
    # This ensures encoder output divides evenly for the adapter reshape
    trunc = conv_len % DOWNSAMPLE_FACTOR
    if trunc > 0:
        h = h[trunc:]
    seq_len = h.shape[0]

    print(f"  Conv stem: {mel.shape[1]} frames -> {conv_len}, left-trunc {trunc} -> {seq_len}", file=sys.stderr)

    # Transformer layers
    positions = torch.arange(seq_len)
    rope_cos, rope_sin = compute_rope_freqs(positions, ENC_HEAD_DIM, ENC_ROPE_THETA)

    for layer in range(ENC_LAYERS):
        lp = f"{prefix}.transformer.layers.{layer}"

        # Pre-attention norm
        attn_norm_w = get_weight(sf_file, f"{lp}.attention_norm.weight")
        norm = RMSNorm(attn_norm_w, ENC_NORM_EPS)
        x_norm = norm(h)

        # Q, K, V (encoder: MHA with biases on q, v, o; no bias on k)
        wq = get_weight(sf_file, f"{lp}.attention.wq.weight")
        wq_b = get_weight(sf_file, f"{lp}.attention.wq.bias")
        wk = get_weight(sf_file, f"{lp}.attention.wk.weight")
        wv = get_weight(sf_file, f"{lp}.attention.wv.weight")
        wv_b = get_weight(sf_file, f"{lp}.attention.wv.bias")
        wo = get_weight(sf_file, f"{lp}.attention.wo.weight")
        wo_b = get_weight(sf_file, f"{lp}.attention.wo.bias")

        # Encoder: NO weight permutation, use is_neox_style=False (per whisper_causal.py)
        q = F.linear(x_norm, wq, wq_b)   # [seq, 2048]
        k = F.linear(x_norm, wk)          # [seq, 2048] no bias
        v = F.linear(x_norm, wv, wv_b)    # [seq, 2048]

        # RoPE - encoder uses is_neox_style=False (interleaved) without weight permutation
        q = apply_rope(q, rope_cos, rope_sin, ENC_HEADS, ENC_HEAD_DIM, is_neox_style=False)
        k = apply_rope(k, rope_cos, rope_sin, ENC_KV_HEADS, ENC_HEAD_DIM, is_neox_style=False)

        # Attention
        attn_out = causal_attention(q, k, v, ENC_HEADS, ENC_KV_HEADS, ENC_HEAD_DIM, ENC_WINDOW)

        # Output projection + residual
        h = h + F.linear(attn_out, wo, wo_b)

        # FFN
        ffn_norm_w = get_weight(sf_file, f"{lp}.ffn_norm.weight")
        ffn_norm = RMSNorm(ffn_norm_w, ENC_NORM_EPS)
        x_norm = ffn_norm(h)

        w1 = get_weight(sf_file, f"{lp}.feed_forward.w1.weight")
        w2 = get_weight(sf_file, f"{lp}.feed_forward.w2.weight")
        w2_b = get_weight(sf_file, f"{lp}.feed_forward.w2.bias")
        w3 = get_weight(sf_file, f"{lp}.feed_forward.w3.weight")

        # SwiGLU: silu(gate) * up, then down
        gate = F.silu(F.linear(x_norm, w1))       # no bias
        up = F.linear(x_norm, w3)                  # no bias
        h = h + F.linear(gate * up, w2, w2_b)      # bias on w2

        if (layer + 1) % 8 == 0:
            print(f"  Encoder layer {layer+1}/{ENC_LAYERS}", file=sys.stderr)

    # Final norm
    final_norm_w = get_weight(sf_file, f"{prefix}.transformer.norm.weight")
    final_norm = RMSNorm(final_norm_w, ENC_NORM_EPS)
    h = final_norm(h)

    print(f"  Encoder final: min={h.min():.4f}, max={h.max():.4f}, mean={h.mean():.4f}", file=sys.stderr)
    return h  # [seq, 1280]

# ============================================================================
# Adapter forward
# ============================================================================

def adapter_forward(enc_out, sf_file):
    """enc_out: [seq, 1280] -> [seq/4, 3072]. seq must be multiple of 4."""
    prefix = "mm_streams_embeddings.embedding_module"
    w0 = get_weight(sf_file, f"{prefix}.audio_language_projection.0.weight")
    w1 = get_weight(sf_file, f"{prefix}.audio_language_projection.2.weight")

    seq_len = enc_out.shape[0]
    assert seq_len % DOWNSAMPLE_FACTOR == 0, f"Encoder output {seq_len} not divisible by {DOWNSAMPLE_FACTOR}"

    # Reshape: [seq, 1280] -> [seq/4, 5120]
    ds = enc_out.reshape(seq_len // DOWNSAMPLE_FACTOR, ENC_DIM * DOWNSAMPLE_FACTOR)

    # MLP: Linear(5120->3072) -> GELU -> Linear(3072->3072)
    out = F.gelu(F.linear(ds, w0))
    out = F.linear(out, w1)

    print(f"  Adapter: {seq_len} -> {out.shape[0]} (downsample {DOWNSAMPLE_FACTOR}x)", file=sys.stderr)
    return out  # [seq/4, 3072]

# ============================================================================
# Decoder forward (prefill + generation)
# ============================================================================

class Decoder:
    def __init__(self, sf_file):
        self.sf = sf_file
        self.dtype = torch.bfloat16 if USE_BF16 else torch.float32
        self.tok_embeddings = get_weight(sf_file,
            "mm_streams_embeddings.embedding_module.tok_embeddings.weight")
        self.final_norm = get_weight(sf_file, "norm.weight")
        self.kv_cache = {}  # layer -> (k_cache, v_cache)

        # Preload layer weights
        self.layers = []
        for i in range(DEC_LAYERS):
            layer = self._load_layer(i)
            self.layers.append(layer)
            if (i + 1) % 8 == 0:
                print(f"  Decoder layer {i+1}/{DEC_LAYERS} loaded", file=sys.stderr)

        if USE_BF16:
            print(f"  Using bfloat16 computation", file=sys.stderr)

    def _load_layer(self, i):
        sf = self.sf
        lp = f"layers.{i}"
        dtype = self.dtype

        # No Q/K permutation needed: safetensors weights are in interleaved format
        # (consecutive pairs form RoPE rotation pairs), matching is_neox_style=False.
        # Verified equivalent to vLLM's maybe_remap_mistral + is_neox_style=True.

        # Large weights in compute dtype, norms stay float32
        return {
            'attention_norm': get_weight(sf, f"{lp}.attention_norm.weight"),  # f32 for precision
            'ffn_norm': get_weight(sf, f"{lp}.ffn_norm.weight"),              # f32 for precision
            'wq': get_weight(sf, f"{lp}.attention.wq.weight").to(dtype),
            'wk': get_weight(sf, f"{lp}.attention.wk.weight").to(dtype),
            'wv': get_weight(sf, f"{lp}.attention.wv.weight").to(dtype),
            'wo': get_weight(sf, f"{lp}.attention.wo.weight").to(dtype),
            'w1': get_weight(sf, f"{lp}.feed_forward.w1.weight").to(dtype),
            'w2': get_weight(sf, f"{lp}.feed_forward.w2.weight").to(dtype),
            'w3': get_weight(sf, f"{lp}.feed_forward.w3.weight").to(dtype),
            # ada_rms_norm_t_cond: Linear(3072->32) -> GELU -> Linear(32->3072)
            # Applied after ffn_norm, before FFN: h = h * (1 + ada_mlp(t_cond))
            'ada_down': get_weight(sf, f"{lp}.ada_rms_norm_t_cond.0.weight").to(dtype),  # [32, 3072]
            'ada_up': get_weight(sf, f"{lp}.ada_rms_norm_t_cond.2.weight").to(dtype),    # [3072, 32]
        }

    def embed_token(self, token_id):
        return self.tok_embeddings[token_id]

    def embed_tokens(self, token_ids):
        """Embed a batch of token IDs. token_ids: 1D tensor of ints."""
        return self.tok_embeddings[token_ids]

    def _layer_forward(self, h, layer_idx, pos, kv_seq_len, t_cond=None, debug=False):
        """Single layer forward for one or more positions.

        t_cond: Time conditioning tensor from TimeEmbedding, shape [3072].
                Applied via ada_rms_norm per layer: h = h * (1 + ada_mlp(t_cond))
        """
        L = self.layers[layer_idx]
        seq_len = h.shape[0]
        dtype = self.dtype

        # Convert input to bf16 if needed
        if USE_BF16 and h.dtype != dtype:
            h = h.to(dtype)

        if debug:
            print(f"    Layer {layer_idx} input: [{h.float().min():.2f}, {h.float().max():.2f}]", file=sys.stderr)

        # Pre-attention RMSNorm (always in float32 for precision)
        norm = RMSNorm(L['attention_norm'], DEC_NORM_EPS)
        x_norm = norm(h)
        if USE_BF16:
            x_norm = x_norm.to(dtype)
        if debug:
            print(f"    After attn_norm: [{x_norm.float().min():.2f}, {x_norm.float().max():.2f}]", file=sys.stderr)

        # Q, K, V (no bias in decoder)
        q = F.linear(x_norm, L['wq'])   # [seq, 4096]
        k = F.linear(x_norm, L['wk'])   # [seq, 1024]
        v = F.linear(x_norm, L['wv'])   # [seq, 1024]

        # RoPE (in float32 for precision) - Interleaved style (is_neox_style=False)
        # Mistral safetensors stores Q/K weights in interleaved format per head:
        # consecutive pairs (q[2j], q[2j+1]) form RoPE rotation pairs.
        # This is equivalent to vLLM's permuted + split-half approach (verified).
        positions = torch.arange(pos, pos + seq_len)
        rope_cos, rope_sin = compute_rope_freqs(positions, DEC_HEAD_DIM, DEC_ROPE_THETA)
        q = apply_rope(q.float(), rope_cos, rope_sin, DEC_HEADS, DEC_HEAD_DIM, is_neox_style=False)
        k = apply_rope(k.float(), rope_cos, rope_sin, DEC_KV_HEADS, DEC_HEAD_DIM, is_neox_style=False)
        if USE_BF16:
            q = q.to(dtype)
            k = k.to(dtype)

        # Update KV cache
        if layer_idx not in self.kv_cache:
            k_cache = k
            v_cache = v
        else:
            k_cache, v_cache = self.kv_cache[layer_idx]
            k_cache = torch.cat([k_cache, k], dim=0)
            v_cache = torch.cat([v_cache, v], dim=0)

        # Keep only last DEC_WINDOW positions
        if k_cache.shape[0] > DEC_WINDOW:
            k_cache = k_cache[-DEC_WINDOW:]
            v_cache = v_cache[-DEC_WINDOW:]

        self.kv_cache[layer_idx] = (k_cache, v_cache)
        full_k, full_v = self.kv_cache[layer_idx]

        # Attention
        kv_start_pos = (pos + seq_len - 1) - (full_k.shape[0] - 1)
        attn_out = causal_attention(
            q, full_k, full_v,
            DEC_HEADS, DEC_KV_HEADS, DEC_HEAD_DIM,
            DEC_WINDOW,
            q_start_pos=pos,
            kv_start_pos=kv_start_pos,
        )

        # Output projection + residual
        attn_proj = F.linear(attn_out, L['wo'])
        if debug:
            print(f"    After attn proj: [{attn_proj.float().min():.2f}, {attn_proj.float().max():.2f}]", file=sys.stderr)
        h = h + attn_proj
        if debug:
            print(f"    After attn residual: [{h.float().min():.2f}, {h.float().max():.2f}]", file=sys.stderr)

        # Pre-FFN RMSNorm
        ffn_norm = RMSNorm(L['ffn_norm'], DEC_NORM_EPS)
        h_norm = ffn_norm(h)
        if USE_BF16:
            h_norm = h_norm.to(dtype)
        if debug:
            print(f"    After ffn_norm: [{h_norm.float().min():.2f}, {h_norm.float().max():.2f}]", file=sys.stderr)

        # Ada RMSNorm time conditioning (applied after ffn_norm, before FFN)
        # h = h * (1 + ada_mlp(t_cond)) where ada_mlp = Linear(3072->32)->GELU->Linear(32->3072)
        if t_cond is not None:
            t_cond_bf16 = t_cond.to(dtype) if USE_BF16 else t_cond
            ada_hidden = F.linear(t_cond_bf16, L['ada_down'])  # [32]
            ada_hidden = F.gelu(ada_hidden)
            ada_scale = F.linear(ada_hidden, L['ada_up'])  # [3072]
            h_norm = h_norm * (1 + ada_scale.unsqueeze(0))  # broadcast over seq_len
            if debug:
                print(f"    After ada_norm: [{h_norm.float().min():.2f}, {h_norm.float().max():.2f}]", file=sys.stderr)

        # SwiGLU FFN
        gate = F.silu(F.linear(h_norm, L['w1']))
        up = F.linear(h_norm, L['w3'])
        ffn_out = F.linear(gate * up, L['w2'])
        if debug:
            print(f"    FFN output: [{ffn_out.float().min():.2f}, {ffn_out.float().max():.2f}]", file=sys.stderr)
        h = h + ffn_out
        if debug:
            print(f"    After FFN residual: [{h.float().min():.2f}, {h.float().max():.2f}]", file=sys.stderr)

        return h

    def prefill(self, input_embeds, t_cond):
        """input_embeds: [seq, 3072], t_cond: [3072] from TimeEmbedding."""
        self.kv_cache = {}
        h = input_embeds
        seq_len = h.shape[0]

        if USE_BF16:
            h = h.to(self.dtype)

        for layer in range(DEC_LAYERS):
            debug_this_layer = (layer < 2)
            h = self._layer_forward(h, layer, 0, seq_len, t_cond=t_cond, debug=debug_this_layer)
            if layer < 4 or (layer + 1) % 8 == 0:
                print(f"  Decoder prefill layer {layer+1}/{DEC_LAYERS}: h range [{h.float().min():.2f}, {h.float().max():.2f}]", file=sys.stderr)

        return h  # [seq, 3072]

    def forward_one(self, embed, pos, t_cond):
        """Generate one token. embed: [1, 3072], t_cond: [3072], returns logits [vocab]."""
        h = embed.unsqueeze(0) if embed.dim() == 1 else embed

        if USE_BF16:
            h = h.to(self.dtype)

        for layer in range(DEC_LAYERS):
            h = self._layer_forward(h, layer, pos, pos + 1, t_cond=t_cond)

        # Final norm (in float32)
        norm = RMSNorm(self.final_norm, DEC_NORM_EPS)
        h = norm(h)

        # Logits via tied embeddings: h @ tok_embeddings^T
        # Always compute logits in float32 for precision
        logits = F.linear(h.float().squeeze(0), self.tok_embeddings)
        return logits  # [vocab]

# ============================================================================
# Tokenizer (minimal: decode token IDs to text)
# ============================================================================

def load_tokenizer(model_dir):
    """Load a minimal Tekken decoder from local `tekken.json` (self-contained).

    We decode token IDs by concatenating per-token `token_bytes` (base64) and
    decoding the resulting byte stream as UTF-8. Control/special tokens are
    skipped.
    """
    tekken_path = os.path.join(model_dir, "tekken.json")
    with open(tekken_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = data["vocab"]
    config = data.get("config", {})
    n_special = int(config.get("default_num_special_tokens", 1000))
    special_ids = {int(st["rank"]) for st in data.get("special_tokens", []) if "rank" in st}

    bytes_cache = {}

    def token_bytes(token_id: int) -> bytes:
        b = bytes_cache.get(token_id)
        if b is not None:
            return b
        if token_id < 0:
            bytes_cache[token_id] = b""
            return b""

        # Model token IDs reserve the first `n_special` IDs for special tokens.
        # Normal vocab tokens are offset by `n_special` into `vocab[]`.
        if token_id < n_special or token_id in special_ids:
            bytes_cache[token_id] = b""
            return b""

        vocab_id = token_id - n_special
        if vocab_id < 0 or vocab_id >= len(vocab):
            bytes_cache[token_id] = b""
            return b""

        b = base64.b64decode(vocab[vocab_id]["token_bytes"])
        bytes_cache[token_id] = b
        return b

    def decode(token_ids):
        out = bytearray()
        for token_id in map(int, token_ids):
            if token_id < n_special or token_id in special_ids:
                continue
            out += token_bytes(token_id)
        return out.decode("utf-8", errors="replace")

    return decode

# ============================================================================
# Full pipeline
# ============================================================================

def transcribe(model_dir, wav_path):
    # Load audio
    audio_array, sr = sf.read(wav_path, dtype='float32')
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        import soxr
        audio_array = soxr.resample(audio_array, sr, SAMPLE_RATE, quality="HQ")

    print(f"Audio: {len(audio_array)} samples ({len(audio_array)/SAMPLE_RATE:.1f}s)", file=sys.stderr)

    # Streaming-format prefix + offline audio padding (self-contained).
    # Prefix tokens: [BOS] + [STREAMING_PAD]*(n_left_pad_tokens + n_delay_tokens)  (len=39 by default)
    prompt_ids = [TOKEN_BOS] + [TOKEN_STREAMING_PAD] * (N_LEFT_PAD_TOKENS + N_DELAY_TOKENS)
    padded = pad_audio_streaming(audio_array).astype(np.float32)
    print(
        f"Tokenizer OFFLINE: prompt_len={len(prompt_ids)} unique={sorted(set(prompt_ids))}",
        file=sys.stderr,
    )
    print(
        f"Audio padded: {len(padded)} samples ({len(padded)/SAMPLE_RATE:.1f}s)",
        file=sys.stderr,
    )

    # Mel spectrogram
    mel_filters = torch.tensor(compute_mel_filters(), dtype=torch.float32)
    audio_tensor = torch.tensor(padded, dtype=torch.float32)
    mel = compute_mel_spectrogram(audio_tensor, mel_filters)
    print(f"Mel: {mel.shape[1]} frames", file=sys.stderr)

    # Truncate left if not divisible by 2 (conv stride)
    if mel.shape[1] % 2 != 0:
        mel = mel[:, 1:]
        print(f"Mel truncated to {mel.shape[1]} frames", file=sys.stderr)

    # Load weights
    sf_path = os.path.join(model_dir, "consolidated.safetensors")
    print(f"Loading model from {sf_path}", file=sys.stderr)
    sf_file = safe_open(sf_path, framework="pt")

    # Encoder
    print("Running encoder...", file=sys.stderr)
    with torch.no_grad():
        enc_out = encoder_forward(mel, None, sf_file)
    print(f"Encoder output: {enc_out.shape}", file=sys.stderr)

    # Adapter (no normalization - matches vendor code)
    print("Running adapter...", file=sys.stderr)
    with torch.no_grad():
        adapter_out = adapter_forward(enc_out, sf_file)
    print(f"Adapter output: {adapter_out.shape}", file=sys.stderr)

    # Load decoder
    print("Loading decoder...", file=sys.stderr)
    decoder = Decoder(sf_file)

    # Time conditioning (ada_rms_norm_t_cond)
    # The decoder uses per-layer adaptive modulation based on `t_cond`.
    t_cond = compute_time_embedding(float(N_DELAY_TOKENS), DEC_DIM)
    print(
        f"Time conditioning: t={N_DELAY_TOKENS}, t_cond shape={t_cond.shape}",
        file=sys.stderr,
    )

    # ----------------------------------------------------------------------
    # Official vLLM realtime decoding schedule (offline WAV)
    #
    # - Prefix: prompt_ids (len=39 by default): BOS + STREAMING_PAD*(left_pad + delay)
    # - Audio positions: one audio embedding per position (adapter_out), length N
    # - Generation happens *within* the audio-token range:
    #   1) Prefill positions [0..L-1] using (audio_embed[pos] + tok_embed(prompt_ids[pos]))
    #   2) Sample next token from last prefix position (pos=L-1) -> token_L
    #   3) For pos=L..N-1:
    #        feed (audio_embed[pos] + tok_embed(prev_token)) and sample next token
    #        stop on EOS
    #
    # This matches vLLM's requirement that a multimodal embedding exists at every step.
    # ----------------------------------------------------------------------

    n_audio = adapter_out.shape[0]
    L = len(prompt_ids)
    assert L > 0, L
    assert L <= n_audio, (L, n_audio)

    prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long)
    prefix_text_embeds = decoder.embed_tokens(prompt_ids_t)  # [L, 3072]
    prefix_embeds = adapter_out[:L] + prefix_text_embeds

    print(f"  audio_tokens={n_audio}, prefix_tokens={L}", file=sys.stderr)
    print(
        f"  adapter_out stats: min={adapter_out.min():.4f}, max={adapter_out.max():.4f}, std={adapter_out.std():.4f}",
        file=sys.stderr,
    )
    print(
        f"  prefix_embeds stats: min={prefix_embeds.min():.4f}, max={prefix_embeds.max():.4f}",
        file=sys.stderr,
    )

    print("Running decoder prefill (prefix)...", file=sys.stderr)
    with torch.no_grad():
        if L > 1:
            _ = decoder.prefill(prefix_embeds[:-1], t_cond)
        logits = decoder.forward_one(prefix_embeds[-1], pos=L - 1, t_cond=t_cond)
        token = int(logits.argmax().item())

    generated = [token]
    print(f"  Token 1 (after prefix): {token}", file=sys.stderr)

    print("Running decoder decode (within audio span)...", file=sys.stderr)
    with torch.no_grad():
        for pos in range(L, n_audio):
            if token == TOKEN_EOS:
                break
            embed = adapter_out[pos] + decoder.embed_token(token)
            logits = decoder.forward_one(embed, pos=pos, t_cond=t_cond)
            token = int(logits.argmax().item())
            generated.append(token)

            if len(generated) <= 5:
                topk_vals, topk_idxs = torch.topk(logits, 5)
                print(
                    f"  Token {len(generated)} (pos={pos}): {token}, top5: {list(zip(topk_idxs.tolist(), topk_vals.tolist()))}",
                    file=sys.stderr,
                )

    print(f"Generated {len(generated)} tokens (raw)", file=sys.stderr)

    # Remove EOS from output
    if generated and generated[-1] == TOKEN_EOS:
        generated = generated[:-1]

    # Decode
    decode = load_tokenizer(model_dir)
    text = decode(generated).strip()

    return text

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <audio.wav>", file=sys.stderr)
        sys.exit(1)

    model_dir = sys.argv[1]
    wav_path = sys.argv[2]

    text = transcribe(model_dir, wav_path)
    print(text)
