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

  it('should sort cloud spatially', () => {
      // Create a small test cloud
      // P1: (10, 0, 0)
      // P2: (0, 0, 0)
      // P3: (5, 0, 0)
      const positions = new Float32Array([
          10, 0, 0,
          0, 0, 0,
          5, 0, 0
      ]);
      const rotations = new Float32Array([0,0,0,1, 0,0,0,1, 0,0,0,1]); // Identity
      const scales = new Float32Array([1, 2, 3]);

      const cloud = { positions, rotations, scales };
      const sorted = E8Lattice.sortCloud(cloud);

      // Expected Order: 0, 5, 10 (Indices 1, 2, 0)
      // Because minX=0, maxX=10.
      // 0 -> code 0
      // 5 -> code 511
      // 10 -> code 1023

      // Check positions
      expect(sorted.positions[0]).toBe(0);
      expect(sorted.positions[3]).toBe(5);
      expect(sorted.positions[6]).toBe(10);

      // Check scales (to verify attribute sync)
      // 0 was index 1 -> scale 2
      // 5 was index 2 -> scale 3
      // 10 was index 0 -> scale 1
      expect(sorted.scales[0]).toBe(2);
      expect(sorted.scales[1]).toBe(3);
      expect(sorted.scales[2]).toBe(1);
  });
});
