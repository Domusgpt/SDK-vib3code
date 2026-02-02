import { describe, it, expect } from 'vitest';
import { PLASTIC_CONSTANT, getPadovanSequence, getPlasticSamplingPoint } from '../../src/math/Plastic';

describe('Plastic Math', () => {
    it('defines the Plastic Constant correctly', () => {
        expect(PLASTIC_CONSTANT).toBeCloseTo(1.324717957244746, 15);
    });

    it('generates the Padovan sequence correctly', () => {
        // Known sequence: 1, 1, 1, 2, 2, 3, 4, 5, 7, 9...
        const seq = getPadovanSequence(10);
        expect(seq).toEqual([1, 1, 1, 2, 2, 3, 4, 5, 7, 9]);
    });

    it('handles edge cases for Padovan sequence', () => {
        expect(getPadovanSequence(0)).toEqual([]);
        expect(getPadovanSequence(1)).toEqual([1]);
        expect(getPadovanSequence(3)).toEqual([1, 1, 1]);
    });

    it('generates low-discrepancy sampling points', () => {
        const p0 = getPlasticSamplingPoint(0);
        expect(p0.x).toBeCloseTo(0.5);
        expect(p0.y).toBeCloseTo(0.5);

        const p1 = getPlasticSamplingPoint(1);
        const rho = PLASTIC_CONSTANT;
        expect(p1.x).toBeCloseTo((0.5 + 1/rho) % 1.0);
        expect(p1.y).toBeCloseTo((0.5 + 1/(rho*rho)) % 1.0);
    });

    it('sampling points stay within [0, 1) range', () => {
        for(let i = 0; i < 100; i++) {
            const p = getPlasticSamplingPoint(i);
            expect(p.x).toBeGreaterThanOrEqual(0);
            expect(p.x).toBeLessThan(1);
            expect(p.y).toBeGreaterThanOrEqual(0);
            expect(p.y).toBeLessThan(1);
        }
    });
});
