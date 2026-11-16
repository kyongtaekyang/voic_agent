// Microphone input (device sample rate) -> mono PCM16 at targetRate, posted to the main thread every chunkMs.
// Simple block-average downsampling (44.1/48 kHz -> 24 kHz). Adapted from korean_agent/web/static/capture-worklet.js.
class Pcm16Capture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = (options && options.processorOptions) || {};
    this.target = o.targetRate || 24000;
    this.ratio = sampleRate / this.target;
    this.chunk = Math.round(this.target * (o.chunkMs || 80) / 1000);
    this.out = new Int16Array(this.chunk);
    this.n = 0;
    this.acc = 0;
    this.cnt = 0;
    this.pos = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.acc += ch[i];
      this.cnt += 1;
      this.pos += 1;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.acc / this.cnt));
        this.out[this.n++] = v < 0 ? v * 0x8000 : v * 0x7fff;
        this.acc = 0;
        this.cnt = 0;
        if (this.n === this.chunk) {
          this.port.postMessage(this.out.buffer.slice(0));
          this.n = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("pcm16-capture", Pcm16Capture);
