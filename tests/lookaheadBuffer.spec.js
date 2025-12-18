import { describe, it, expect } from 'vitest';
import { LookaheadBuffer } from '../src/utils/lookaheadBuffer.js';

describe('LookaheadBuffer', () => {
  it('returns projected frame when future data is unavailable', () => {
    const buffer = new LookaheadBuffer({ lookaheadMs: 200 });
    const now = 1000;

    buffer.pushFrame([0.1, 0.2, 0.3], 0.4, now);
    buffer.pushFrame([0.2, 0.3, 0.4], 0.5, now + 50);

    const frame = buffer.getProjectedFrame(now + 100);
    expect(frame.projected).toBe(true);
    expect(frame.bands[0]).toBeGreaterThanOrEqual(0.2);
    expect(frame.energy).toBeGreaterThanOrEqual(0.5);
  });

  it('returns real future frame when available', () => {
    const buffer = new LookaheadBuffer({ lookaheadMs: 200 });
    const now = 2000;

    buffer.pushFrame([0.2, 0.2, 0.2], 0.2, now);
    buffer.pushFrame([0.3, 0.3, 0.3], 0.3, now + 250);

    const frame = buffer.getProjectedFrame(now + 20);
    expect(frame.projected).toBe(false);
    expect(frame.timestamp).toBeGreaterThanOrEqual(now + 200);
  });

  it('clamps bands and energy values', () => {
    const buffer = new LookaheadBuffer();
    buffer.pushFrame([2, -1], 5, 0);
    const frame = buffer.getProjectedFrame(0);
    expect(frame.bands.every((val) => val >= 0 && val <= 1)).toBe(true);
    expect(frame.energy).toBeLessThanOrEqual(1.5);
  });
});

