import { describe, it, expect } from 'vitest';
import { E8Lattice } from '../../src/math/E8.js';

describe('E8Lattice', () => {
  it('should project 3D to 8D', () => {
    const v8 = E8Lattice.project3Dto8D(1, 1, 1);
    expect(v8).toBeInstanceOf(Float32Array);
    expect(v8.length).toBe(8);
    // Check specific values based on phi
    const phi = 1.61803398875;
    expect(v8[0]).toBeCloseTo(1);
    expect(v8[3]).toBeCloseTo(phi);
  });

  it('should generate 240 roots', () => {
    const roots = E8Lattice.generateRoots();
    expect(roots.length).toBe(240);

    // Check norm squared of the first root.
    // Roots are either (±1, ±1, 0...) -> length^2 = 2
    // Or (±0.5...) -> length^2 = 8 * 0.25 = 2.
    const r0 = roots[0];
    let normSq = 0;
    for(let i=0; i<8; i++) normSq += r0[i]*r0[i];
    expect(normSq).toBeCloseTo(2);
  });

  it('should project 8D to 3D', () => {
    const v8 = new Float32Array([1, 0, 0, 0, 0, 0, 0, 0]);
    const v3 = E8Lattice.project8Dto3D(v8);
    expect(v3).toHaveProperty('x');
    expect(v3).toHaveProperty('y');
    expect(v3).toHaveProperty('z');
  });

  it('should fold 8D to 4D (Moxness)', () => {
    const v8 = new Float32Array([1, 1, 1, 1, 1, 1, 1, 1]);
    const { left, right } = E8Lattice.fold(v8);
    expect(left).toBeInstanceOf(Float32Array);
    expect(right).toBeInstanceOf(Float32Array);
    expect(left.length).toBe(4);
    expect(right.length).toBe(4);
  });

  it('should quantize to lattice (D8)', () => {
      // Test D8 rounding
      // (2.1, 0, ...) -> (2, 0, ...)
      // Sum = 2 (even) -> OK.
      const v8 = new Float32Array([2.1, 0.1, 0, 0, 0, 0, 0, 0]);
      const result = E8Lattice.quantizeToLattice(v8);
      expect(result.index[0]).toBe(2);
      expect(result.index[1]).toBe(0);

      // Test odd sum correction
      // (1.1, 0, ...) -> (1, 0...) sum=1 (odd).
      // Max error is at index 0 (0.1).
      // Should adjust to 0 or 2?
      // 1.1 -> round 1. err 0.1.
      // If we change 1->2, err is 0.9. If we change 1->0, err is 1.1.
      // Wait, rounding logic:
      // if v > round, round += 1. 1.1 > 1, so becomes 2.
      // (2, 0...) sum=2. Correct.
       const v8_odd = new Float32Array([1.1, 0.1, 0, 0, 0, 0, 0, 0]);
       const result_odd = E8Lattice.quantizeToLattice(v8_odd);
       // The logic is: finds max error.
       // round(1.1) = 1. err=0.1.
       // round(0.1) = 0. err=0.1.
       // maxErrIdx could be 0.
       // if (v[0] > round[0]) -> 1.1 > 1 -> round[0] += 1 -> 2.
       expect(result_odd.index[0]).toBe(2);
  });
});
