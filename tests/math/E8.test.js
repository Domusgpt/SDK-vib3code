import { describe, it, expect } from 'vitest';
import { E8Lattice } from '../../src/math/E8.js';

describe('E8Lattice (Quatossian)', () => {

  it('should generate 240 roots', () => {
    const roots = E8Lattice.generateRoots();
    // Flat array: 240 * 8
    expect(roots.length).toBe(240 * 8);

    // Check first root length (should be sqrt(2))
    // Roots are (±1, ±1, 0...) or (±0.5...)
    let normSq = 0;
    for(let i=0; i<8; i++) normSq += roots[i]*roots[i];
    expect(normSq).toBeCloseTo(2.0);
  });

  it('should fold 8D to 4D (Moxness)', () => {
    const v8 = new Float32Array([1, 1, 0, 0, 0, 0, 0, 0]); // A root
    const q = E8Lattice.foldTo4D(v8);
    expect(q.length).toBe(4);
    // Check values are not NaN
    expect(q[0]).not.toBeNaN();
  });

  it('should generate Quatossian Cloud', () => {
    const shells = 2;
    const cloud = E8Lattice.generateCloud(shells);

    const numPoints = 240 * shells;
    expect(cloud.positions.length).toBe(numPoints * 3);
    expect(cloud.rotations.length).toBe(numPoints * 4);

    // Verify Quaternion normalization
    const q = cloud.rotations.subarray(0, 4);
    const len = Math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    expect(len).toBeCloseTo(1.0);
  });
});
