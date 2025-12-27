import { describe, it, expect } from 'vitest';
import { SequenceDirector } from '../src/choreo/SequenceDirector.js';

const director = new SequenceDirector();

describe('SequenceDirector', () => {
  it('boosts highlight/accent when snap surpasses gate', () => {
    const result = director.evaluate(
      {
        triBands: new Float32Array([0.1, 0.2, 0.5]),
        energy: 0.3,
        projected: false,
        sequence: { heartbeat: 0.1, swell: 0.12, snap: 0.35 }
      },
      { player_health: 0.9, combo_multiplier: 1, zone: 'combat' }
    );

    expect(result.layerAlpha.highlight).toBeGreaterThan(1.1);
    expect(result.layerAlpha.accent).toBeGreaterThan(1.0);
    expect(result.geometry).toBe('tesseract');
    expect(result.strokeWidth).toBeGreaterThan(1.5);
  });

  it('respects anticipation when projected frame exists', () => {
    const result = director.evaluate(
      {
        triBands: new Float32Array([0.05, 0.05, 0.05]),
        energy: 0.1,
        projected: true,
        sequence: { heartbeat: 0.2, swell: 0.2, snap: 0.2 }
      },
      { player_health: 1, combo_multiplier: 0 }
    );

    expect(result.layerAlpha.background).toBeGreaterThan(0.95);
    expect(result.layerAlpha.content).toBeGreaterThan(1.0);
    expect(result.projected).toBe(true);
  });

  it('clamps stroke width within configured bounds', () => {
    const result = director.evaluate(
      {
        triBands: new Float32Array([0.5, 0.5, 0.5]),
        energy: 1,
        projected: false,
        sequence: { heartbeat: 1, swell: 1, snap: 1 }
      },
      { player_health: 0, combo_multiplier: 10 }
    );

    expect(result.strokeWidth).toBeLessThanOrEqual(director.stroke.max);
    expect(result.strokeWidth).toBeGreaterThanOrEqual(director.stroke.min);
  });
});
