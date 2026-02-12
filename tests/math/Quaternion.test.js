import { describe, it, expect } from 'vitest';
import { Quaternion } from '../../src/math/Quaternion.js';

describe('Quaternion', () => {
    it('should create a quaternion', () => {
        const q = new Quaternion(0, 0, 0, 1);
        expect(q.w).toBe(1);
    });

    it('should calculate Hamilton product (multiply)', () => {
        const a = new Quaternion(0, 0, 0, 1); // Identity
        const b = new Quaternion(1, 0, 0, 0); // i
        const c = Quaternion.multiply(a, b);
        expect(c.x).toBe(1);
        expect(c.w).toBe(0);
    });

    it('should rotate 4D vector (left multiplication)', () => {
        // Test i * j = k
        // q = i, v = j
        const i = new Quaternion(1, 0, 0, 0);
        const j = new Float32Array([0, 1, 0, 0]);

        // q * v
        const res = Quaternion.rotate4DLeft(j, i);

        // Expect k (0, 0, 1, 0)
        expect(res[0]).toBe(0);
        expect(res[1]).toBe(0);
        expect(res[2]).toBe(1);
        expect(res[3]).toBe(0);
    });
});
