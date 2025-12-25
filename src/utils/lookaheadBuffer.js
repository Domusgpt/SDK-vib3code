class LookaheadBuffer {
  constructor({ lookaheadMs = 200, maxFrames = 180 } = {}) {
    this.lookaheadMs = lookaheadMs;
    this.maxFrames = maxFrames;
    this.frames = [];
  }

  pushFrame(bands, energy, timestamp = performance.now(), extras = undefined) {
    if (!Array.isArray(bands) && !ArrayBuffer.isView(bands)) return;
    const normalizedBands = Array.from(bands, (value) => Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0);
    const clampedEnergy = Number.isFinite(energy) ? Math.max(0, Math.min(1.5, energy)) : 0;

    this.frames.push({ bands: normalizedBands, energy: clampedEnergy, timestamp, extras });
    if (this.frames.length > this.maxFrames) {
      this.frames.shift();
    }
  }

  getProjectedFrame(now = performance.now()) {
    if (!this.frames.length) return null;

    const targetTime = now + this.lookaheadMs;
    const futureFrame = this.frames.find((frame) => frame.timestamp >= targetTime);
    if (futureFrame) return { ...futureFrame, projected: false };

    const latest = this.frames[this.frames.length - 1];
    const previous = this.frames[this.frames.length - 2] || latest;
    const dt = Math.max(1, latest.timestamp - previous.timestamp);
    const factor = Math.max(0, Math.min(2, (targetTime - latest.timestamp) / dt));

    const projectedBands = latest.bands.map((value, idx) => {
      const delta = value - (previous.bands?.[idx] ?? value);
      const estimate = value + delta * factor;
      return Math.max(0, Math.min(1, Number(estimate.toFixed(4))));
    });

    const energyDelta = latest.energy - (previous.energy ?? latest.energy);
    const projectedEnergy = Math.max(0, Math.min(1.5, Number((latest.energy + energyDelta * factor).toFixed(4))));

    return {
      bands: projectedBands,
      energy: projectedEnergy,
      timestamp: targetTime,
      projected: true,
      extras: latest.extras
    };
  }
}

export { LookaheadBuffer };
