import { normalizeBandsFromFFT } from '../utils/audioBands.js';
import { getDefaultBands, updateBandState } from '../utils/bandSmoothing.js';
import { LookaheadBuffer } from '../utils/lookaheadBuffer.js';
import { computeTriBands, deriveSequenceState } from '../choreo/triBandAnalysis.js';

class AudioChoreographer {
  constructor({ lookaheadMs = 200, fftSize = 256, smoothing = 0.8 } = {}) {
    this.config = { lookaheadMs, fftSize, smoothing };
    this.context = null;
    this.analyser = null;
    this.fftBuffer = null;
    this.sourceCleanup = null;
    this.lookahead = new LookaheadBuffer({ lookaheadMs });
    this.sequenceMode = 'hybrid';
    this.previousTriBands = getDefaultBands(3);

    this.state = {
      bands: getDefaultBands(7),
      triBands: getDefaultBands(3),
      energy: 0,
      projected: false,
      telemetry: { player_health: 1, combo_multiplier: 0, zone: 'calm' },
      sequence: { heartbeat: 0, swell: 0, snap: 0, intensity: 0, mode: this.sequenceMode }
    };
  }

  async start({ preferMic = true } = {}) {
    if (preferMic) {
      try {
        await this.startWithMicrophone();
        return;
      } catch (error) {
        console.warn('Mic unavailable, falling back to oscillator', error);
      }
    }
    await this.startWithOscillator();
  }

  async startWithMicrophone() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const ctx = this.getContext();
    const source = ctx.createMediaStreamSource(stream);
    this.attachAnalyser(ctx, source);
    this.sourceCleanup = () => stream.getTracks().forEach((t) => t.stop());
  }

  async startWithOscillator() {
    const ctx = this.getContext();
    const osc = ctx.createOscillator();
    osc.type = 'triangle';
    osc.frequency.value = 180;
    const gain = ctx.createGain();
    gain.gain.value = 0.05;
    osc.connect(gain);
    this.attachAnalyser(ctx, gain);
    osc.start();
    this.sourceCleanup = () => {
      osc.stop();
      osc.disconnect();
      gain.disconnect();
    };
  }

  async startWithFile(file) {
    const ctx = this.getContext();
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
    const bufferSource = ctx.createBufferSource();
    bufferSource.buffer = audioBuffer;
    bufferSource.loop = true;
    bufferSource.start(0);

    const gain = ctx.createGain();
    gain.gain.value = 0.9;
    bufferSource.connect(gain);

    this.attachAnalyser(ctx, gain);
    this.sourceCleanup = () => {
      bufferSource.stop();
      bufferSource.disconnect();
      gain.disconnect();
    };
  }

  getContext() {
    if (!this.context) {
      this.context = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (this.context.state === 'suspended') {
      this.context.resume();
    }
    return this.context;
  }

  attachAnalyser(ctx, source) {
    if (this.sourceCleanup) this.sourceCleanup();
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = this.config.fftSize;
    this.analyser.smoothingTimeConstant = this.config.smoothing;
    source.connect(this.analyser);
    this.fftBuffer = new Uint8Array(this.analyser.frequencyBinCount);
  }

  setTelemetry(payload = {}) {
    this.state.telemetry = { ...this.state.telemetry, ...payload };
  }

  setSequenceMode(mode = 'hybrid') {
    this.sequenceMode = mode;
    this.state.sequence.mode = mode;
  }

  sampleFrame(now = performance.now()) {
    if (!this.analyser || !this.fftBuffer) return null;
    this.analyser.getByteFrequencyData(this.fftBuffer);

    const bands7 = normalizeBandsFromFFT(this.fftBuffer, this.context?.sampleRate || 48000);
    const triBands = computeTriBands(this.fftBuffer);

    const smoothing = Math.max(0.05, Math.min(1, this.config.smoothing));
    const { bands: smoothed7, energy } = updateBandState(this.state.bands, bands7, smoothing);
    const { bands: smoothedTri } = updateBandState(this.state.triBands, triBands, smoothing);

    this.lookahead.pushFrame(smoothed7, energy, now, { triBands: smoothedTri });
    const projected = this.lookahead.getProjectedFrame(now);

    const nextBands = projected ? Float32Array.from(projected.bands) : smoothed7;
    const nextTri = projected?.extras?.triBands ? Float32Array.from(projected.extras.triBands) : smoothedTri;
    const nextEnergy = projected ? projected.energy : energy;

    const sequence = deriveSequenceState(nextTri, this.previousTriBands, this.sequenceMode);
    this.previousTriBands = nextTri;

    this.state = {
      ...this.state,
      bands: nextBands,
      triBands: nextTri,
      energy: nextEnergy,
      projected: Boolean(projected?.projected),
      sequence
    };

    return this.state;
  }

  stop() {
    if (this.sourceCleanup) {
      this.sourceCleanup();
      this.sourceCleanup = null;
    }
    if (this.context) {
      this.context.close();
      this.context = null;
    }
    this.analyser = null;
    this.fftBuffer = null;
  }
}

export { AudioChoreographer };
