import { describe, it, expect } from 'vitest';
import { JSOH } from '../../src/export/JSOH.js';
import { PLASTIC_CONSTANT } from '../../src/math/Plastic.js';

describe('JSOH Generator', () => {
    it('generates correct schema structure', () => {
        const data = {
            positions: new Float32Array([0,0,0, 1,1,1]),
            scales: new Float32Array([1, 0.5]),
            colors: new Float32Array([1,0,0, 0,1,0])
        };

        const result = JSOH.generate(data);

        expect(result.schema).toBe('JSOH/1.0');
        expect(result.metadata.plasticRatio).toBe(PLASTIC_CONSTANT);
        expect(result.metadata.splatCount).toBe(2);

        // Verify centroid (average of 0,0,0 and 1,1,1 is 0.5,0.5,0.5)
        expect(result.semanticMassing.centroid[0]).toBeCloseTo(0.5);
        expect(result.semanticMassing.centroid[1]).toBeCloseTo(0.5);
        expect(result.semanticMassing.centroid[2]).toBeCloseTo(0.5);

        // Verify bounds
        expect(result.semanticMassing.bounds.min).toEqual([0,0,0]);
        expect(result.semanticMassing.bounds.max).toEqual([1,1,1]);

        // Colors should be hex (assuming 0-1 mapped to 0-255)
        // 1,0,0 -> 255,0,0 -> ff0000
        expect(result.data.colors[0]).toBe('ff0000');
        // 0,1,0 -> 0,255,0 -> 00ff00
        expect(result.data.colors[1]).toBe('00ff00');
    });

    it('processes 100k points efficiently', () => {
        const count = 100000;
        const positions = new Float32Array(count * 3);
        const scales = new Float32Array(count);
        const colors = new Float32Array(count * 3);

        // Fill with dummy data
        for(let i=0; i<count; i++) {
            positions[i*3] = Math.random();
            scales[i] = Math.random();
            colors[i*3] = Math.random();
        }

        const start = performance.now();
        const result = JSOH.generate({ positions, scales, colors });
        const end = performance.now();

        // Ensure it ran correctly
        expect(result.metadata.splatCount).toBe(count);
        expect(result.semanticMassing.bounds.min[0]).toBeLessThan(1);

        // Performance check: 100k points should take less than 200ms on most machines
        // (Just calculating min/max/centroid + hex conversion)
        console.log(`JSOH Generation for 100k points: ${(end-start).toFixed(2)}ms`);
        expect(end - start).toBeLessThan(1000);
    });
});
