// 마이크 입력(브라우저 기본 샘플레이트)을 16kHz 모노 PCM16 으로 바꿔 약 64ms 단위로 메인 스레드에 보낸다.
// 간단한 구간 평균 방식으로 줄인다 (44.1/48kHz → 16kHz). 음성 인식용으로 충분하다.
class Pcm16Capture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.target = (options.processorOptions && options.processorOptions.targetRate) || 16000;
    this.ratio = sampleRate / this.target;
    this.chunk = Math.round(this.target * 0.064);
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
