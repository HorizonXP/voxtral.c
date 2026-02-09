/* AudioWorklet processor to downsample mono float32 audio to 16kHz PCM16.
 *
 * This is intentionally simple (linear interpolation) and designed for
 * realtime dictation, not high-fidelity audio processing.
 */

class VoxtralPCM16Downsample extends AudioWorkletProcessor {
  constructor() {
    super();
    this.outRate = 16000;
    this.ratio = sampleRate / this.outRate;
    // Position (in input samples) where the next output sample should be taken.
    // Always kept in [0, ratio) after each block to avoid negative indexing.
    this.pos = 0.0;
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input || input.length === 0) return true;

    const ratio = this.ratio;
    const inLen = input.length;
    let pos = this.pos;

    // How many output samples fit in this input block?
    // Use ceil (not floor) so we don't carry a negative position into the next block.
    const outLen = Math.max(0, Math.ceil((inLen - pos) / ratio));
    if (outLen === 0) return true;

    const out = new Int16Array(outLen);
    for (let i = 0; i < outLen; i++) {
      // Clamp idx to avoid edge-case float rounding pushing us past the buffer.
      const idx = Math.min(inLen - 1, pos + i * ratio);
      const i0 = Math.floor(idx);
      const i1 = Math.min(i0 + 1, inLen - 1);
      const frac = idx - i0;
      const s = input[i0] + (input[i1] - input[i0]) * frac;
      const v = Math.max(-1.0, Math.min(1.0, s));
      out[i] = v < 0 ? (v * 32768) : (v * 32767);
    }

    // Advance cursor and wrap into next block.
    pos = pos + outLen * ratio - inLen;
    if (pos < 0) pos = 0; // Shouldn't happen, but stay safe under float drift.
    this.pos = pos;

    // Transfer ownership of the buffer to avoid copies.
    this.port.postMessage(out.buffer, [out.buffer]);
    return true;
  }
}

registerProcessor("voxtral-pcm16-downsample", VoxtralPCM16Downsample);
