import { describe, it, expect } from 'vitest';
import { computeTriBands, deriveSequenceState } from '../src/choreo/triBandAnalysis.js';

describe('triBandAnalysis', () => {
  it('aggregates 128-bin FFT data into bass/mid/high averages', () => {
    const fft = new Uint8Array(128).fill(0);
    fft.fill(200, 0, 11); // bass
    fft.fill(100, 11, 101); // mid
    fft.fill(240, 101, 128); // high

    const [bass, mid, high] = computeTriBands(fft);
    expect(bass).toBeGreaterThan(mid);
    expect(high).toBeGreaterThan(mid);
    expect(high).toBeGreaterThan(bass);
  });

  it('derives sequence intensities with mode gating and deltas', () => {
    const previous = new Float32Array([0.1, 0.1, 0.1]);
    const current = new Float32Array([0.5, 0.3, 0.5]);
    const hybrid = deriveSequenceState(current, previous, 'hybrid');
    const character = deriveSequenceState(current, previous, 'character');

    expect(hybrid.heartbeat).toBeGreaterThan(0.2);
    expect(hybrid.snap).toBeGreaterThan(0.2);
    expect(character.snap).toBeGreaterThan(hybrid.snap);
    expect(hybrid.intensity).toBeGreaterThan(0.2);
  });
});

