const DEFAULT_TRI_BAND_RANGES = [
  { start: 0, end: 10 },
  { start: 11, end: 100 },
  { start: 101, end: -1 }
];

function resolveRangeEnd(end, total) {
  if (end < 0) return total - 1;
  return Math.min(end, total - 1);
}

function computeTriBands(fft, ranges = DEFAULT_TRI_BAND_RANGES) {
  if (!fft?.length) return new Float32Array([0, 0, 0]);

  const results = new Float32Array(3);
  const counts = [0, 0, 0];
  const totalBins = fft.length;

  ranges.forEach((range, idx) => {
    const start = Math.max(0, Math.min(totalBins - 1, range.start));
    const end = resolveRangeEnd(range.end, totalBins);
    const clampedEnd = Math.max(start, end);

    for (let i = start; i <= clampedEnd; i++) {
      results[idx] += (fft[i] ?? 0) / 255;
      counts[idx] += 1;
    }

    const count = Math.max(1, counts[idx]);
    results[idx] = Number((results[idx] / count).toFixed(4));
  });

  return results;
}

function deriveSequenceState(currentBands, previousBands = [0, 0, 0], mode = 'hybrid') {
  const len = Math.max(currentBands?.length || 0, 3);
  const current = new Float32Array(len);
  const prev = new Float32Array(len);

  for (let i = 0; i < len; i++) {
    current[i] = currentBands?.[i] ?? 0;
    prev[i] = previousBands?.[i] ?? 0;
  }

  const deltas = current.map((value, idx) => Math.max(0, value - prev[idx]));
  const gain = mode === 'environment' ? [1.05, 0.85, 0.9] : mode === 'character' ? [0.9, 0.95, 1.2] : [1, 1, 1];

  const heartbeat = Math.max(0, Math.min(1, current[0] * gain[0] + deltas[0] * 0.6));
  const swell = Math.max(0, Math.min(1, current[1] * gain[1] + deltas[1] * 0.4));
  const snap = Math.max(0, Math.min(1, current[2] * gain[2] + deltas[2] * 0.9));

  const gate = (value, threshold) => (value >= threshold ? Number(value.toFixed(4)) : 0);

  return {
    mode,
    heartbeat: gate(heartbeat, 0.22),
    swell: gate(swell, 0.18),
    snap: gate(snap, 0.2),
    intensity: Number(Math.max(heartbeat, swell, snap).toFixed(4)),
    deltas
  };
}

export { DEFAULT_TRI_BAND_RANGES, computeTriBands, deriveSequenceState };
