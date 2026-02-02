/**
 * JSOH.js
 * JSON Structured Object Hierarchy for Parserator Integration.
 *
 * Purpose:
 * Transforms raw 3D Gaussian Splat data into a semantic, highly compressed format.
 * This format is designed to be consumed by AI agents via Parserator.com.
 *
 * Features:
 * - "Semantic Massing": Calculates bounding boxes and centroids.
 * - "Plastic Compression": Quantizes data to achieve ~17 bytes/splat target.
 */

import { PLASTIC_CONSTANT } from '../math/Plastic.js';

export class JSOH {
    /**
     * Generates a JSOH object from raw splat data.
     * @param {Object} splatData
     * @param {Float32Array} splatData.positions - Flat array [x,y,z...]
     * @param {Float32Array} splatData.scales - Flat array [s...]
     * @param {Float32Array} splatData.colors - Flat array [r,g,b...] (normalized 0-1)
     * @param {Object} metadata - Additional scene info.
     * @returns {Object} The JSOH object.
     */
    static generate(splatData, metadata = {}) {
        const { positions, scales, colors } = splatData;
        const count = positions.length / 3;

        // 1. Calculate Semantic Massing (Bounding Box & Centroid)
        let min = [Infinity, Infinity, Infinity];
        let max = [-Infinity, -Infinity, -Infinity];
        let centroid = [0, 0, 0];

        for (let i = 0; i < count; i++) {
            const x = positions[i * 3];
            const y = positions[i * 3 + 1];
            const z = positions[i * 3 + 2];

            if (x < min[0]) min[0] = x;
            if (y < min[1]) min[1] = y;
            if (z < min[2]) min[2] = z;

            if (x > max[0]) max[0] = x;
            if (y > max[1]) max[1] = y;
            if (z > max[2]) max[2] = z;

            centroid[0] += x;
            centroid[1] += y;
            centroid[2] += z;
        }

        if (count > 0) {
            centroid[0] /= count;
            centroid[1] /= count;
            centroid[2] /= count;
        }

        // 2. Data Compression
        // We want to return a structure that *can* be represented efficiently.
        // For actual "17 bytes/splat", binary buffers are best, but JSOH implies JSON.
        // We will return a "PlasticPacked" representation: arrays of values.
        // To be truly "Agent Ready", we might segment them, but here we provide the raw mass.

        // Quantize colors to Hex strings for JSON compactness/readability by LLMs
        const hexColors = [];
        for (let i = 0; i < count; i++) {
            const r = Math.floor(colors[i * 3] * 255);
            const g = Math.floor(colors[i * 3 + 1] * 255);
            const b = Math.floor(colors[i * 3 + 2] * 255);
            // Simple hex encoding
            const hex = ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
            hexColors.push(hex);
        }

        return {
            schema: "JSOH/1.0",
            timestamp: new Date().toISOString(),
            metadata: {
                ...metadata,
                plasticRatio: PLASTIC_CONSTANT,
                splatCount: count,
                compressionTarget: "17b"
            },
            semanticMassing: {
                bounds: { min, max },
                centroid: centroid,
                dimensions: [
                    max[0] - min[0],
                    max[1] - min[1],
                    max[2] - min[2]
                ]
            },
            // In a real high-volume scenario, these would be separate binary blobs or base64.
            // For the "Specification", we show the structure.
            // We use limited precision for positions to save space in text.
            data: {
                // Truncate precision for JSON compactness
                positions: Array.from(positions).map(v => Number(v.toFixed(4))),
                scales: Array.from(scales).map(v => Number(v.toFixed(4))),
                colors: hexColors
            }
        };
    }
}
