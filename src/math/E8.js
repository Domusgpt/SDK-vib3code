/**
 * E8.js
 * The Mathematical Core of the Quatossian Inscription Framework.
 *
 * Provides:
 * 1. Generation of E8 Gosset Lattice Roots (240 vertices of 4_21 polytope).
 * 2. Moxness Folding: A projection from 8D -> 4D (H4 600-cell) -> 3D.
 * 3. Quatossian Kernel: Lattice quantization and quaternion spin generation.
 *
 * References:
 * - J.G. Moxness, "The E8 Polytopes"
 * - Conway & Sloane, "Sphere Packings, Lattices and Groups"
 */

import { PLASTIC_CONSTANT } from './Plastic.js';

export class E8Lattice {

    // The Golden Ratio
    static PHI = 1.61803398875;
    static INV_PHI = 0.61803398875; // 1/phi = phi - 1

    /**
     * Generates the 240 Roots of the E8 Lattice.
     * These are the vertices of the Gosset 4_21 polytope.
     * Length squared = 2.
     *
     * Structure:
     * - 112 roots: Permutations of (±1, ±1, 0^6)
     * - 128 roots: (±1/2)^8 with even number of minus signs
     *
     * @returns {Float32Array} Flat array of 240 * 8 coordinates.
     */
    static generateRoots() {
        const roots = [];

        // 1. Integer Roots (112)
        // Permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
        // Indices j < k
        for (let j = 0; j < 8; j++) {
            for (let k = j + 1; k < 8; k++) {
                // Four sign combinations: (+,+), (+,-), (-,+), (-,-)
                // But order matters for the vector, so (1,1) at (0,1) is different from...
                // Wait, these are coordinates.
                // v[j] and v[k] are set.

                const signs = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
                for (let s of signs) {
                    const v = new Float32Array(8);
                    v[j] = s[0];
                    v[k] = s[1];
                    roots.push(v);
                }
            }
        }

        // 2. Half-Integer Roots (128)
        // (±1/2, ..., ±1/2) with even sum of signs
        // We iterate 0..255 (binary)
        for (let i = 0; i < 256; i++) {
            // Check parity (number of set bits)
            // We want even number of MINUS signs.
            // Let's say bit 1 = minus (-0.5), bit 0 = plus (+0.5).

            let minusCount = 0;
            let temp = i;
            while (temp > 0) {
                if (temp & 1) minusCount++;
                temp >>= 1;
            }

            if (minusCount % 2 === 0) {
                const v = new Float32Array(8);
                for (let b = 0; b < 8; b++) {
                    v[b] = ((i >> b) & 1) ? -0.5 : 0.5;
                }
                roots.push(v);
            }
        }

        // Flatten
        const flat = new Float32Array(roots.length * 8);
        for (let i = 0; i < roots.length; i++) {
            flat.set(roots[i], i * 8);
        }
        return flat;
    }

    /**
     * Projects 8D E8 coordinates to 4D H4 (600-cell) space.
     * This is the "Moxness Folding" operation.
     *
     * The projection matrix typically folds E8 using the Golden Ratio to preserve H4 symmetry.
     * Columns of the projection matrix:
     * C_i = c_i * (1, phi) ... (simplified concept)
     *
     * We use the standard folding:
     * u = (v1 + phi*v5, v2 + phi*v6, v3 + phi*v7, v4 + phi*v8) * factor
     *
     * @param {Float32Array} v8 - 8D Vector
     * @returns {Float32Array} 4D Vector (Quaternion)
     */
    static foldTo4D(v8) {
        const phi = E8Lattice.PHI;
        // Scale factor to normalize the resulting 4D roots to a standard size (e.g. 1 or 2)
        // E8 roots length^2 = 2.
        // H4 roots (600-cell) include permutations of (±1, 0, 0, 0), (±1/2, ±1/2, ±1/2, ±1/2), (0, ±1/2, ±phi/2, ±1/2phi)...

        // This linear combination (a + phi*b) is a standard projection from E8 to H4.
        const q = new Float32Array(4);

        // We pair dimensions (0,4), (1,5), (2,6), (3,7)
        // This is a specific chosen basis that works for the standard E8 definition.

        q[0] = v8[0] + phi * v8[4];
        q[1] = v8[1] + phi * v8[5];
        q[2] = v8[2] + phi * v8[6];
        q[3] = v8[3] + phi * v8[7];

        // The "Shadow" is the other fold: v_i - (1/phi)*v_{i+4}

        return q;
    }

    /**
     * Projects 4D (Quaternion) to 3D Space.
     *
     * @param {Float32Array} q - 4D Vector
     * @returns {Float32Array} 3D Vector
     */
    static project4Dto3D(q) {
        // Standard stereographic or orthographic projection
        // For the "Moxness" shadow, we often just drop W or do a perspective divide.

        // Let's use a perspective projection from 4D to 3D to see the "Shadow".
        // w is the "time" or "scale" dimension.

        const w = q[3];
        const dist = 2.0 - w; // Camera at w=2
        const f = 1.0 / (dist > 0.001 ? dist : 0.001);

        return new Float32Array([
            q[0] * f,
            q[1] * f,
            q[2] * f
        ]);
    }

    /**
     * Generates a "Quatossian Cloud".
     * A set of points in 3D, each with an associated Quaternion spin.
     *
     * @param {number} shells - Number of lattice shells to generate (approx radius).
     * @returns {Object} { positions: Float32Array, rotations: Float32Array }
     */
    static generateCloud(shells = 1) {
        // For efficiency in JS, we'll just use the Roots (Shell 1) and scale them
        // using the Plastic Ratio to create "Phi-shells".
        // This mimics the "Quasicrystal" growth.

        const rootsFlat = this.generateRoots();
        const numRoots = 240;

        const count = numRoots * shells;
        const positions = new Float32Array(count * 3);
        const rotations = new Float32Array(count * 4); // Quaternions

        const phi = E8Lattice.PHI;

        let idx = 0;

        for (let s = 0; s < shells; s++) {
            // Scale grows by Plastic Constant or Phi?
            // "Plastic Ratio (rho ~ 1.3247) Sampling"
            const scale = Math.pow(PLASTIC_CONSTANT, s + 1);

            for (let i = 0; i < numRoots; i++) {
                // Extract 8D root
                const v8 = rootsFlat.subarray(i*8, i*8+8);

                // Fold to 4D (This is our "Spin" state!)
                // The 4D projection of an E8 root is a quaternion.
                const q = this.foldTo4D(v8);

                // Normalize q to be a valid rotation quaternion
                let len = Math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
                if (len > 0) {
                    q[0]/=len; q[1]/=len; q[2]/=len; q[3]/=len;
                }

                // Store Rotation
                rotations[idx * 4 + 0] = q[0];
                rotations[idx * 4 + 1] = q[1];
                rotations[idx * 4 + 2] = q[2];
                rotations[idx * 4 + 3] = q[3];

                // Project to 3D (Position)
                // We use the 4D point projected to 3D, scaled by the shell factor.
                const p3 = this.project4Dto3D(q); // Re-calculating un-normalized 4D? No, use normalized direction * scale?
                // Actually, the lattice points are at specific radii.
                // Let's map the lattice point to 3D directly.

                // Position = Direction of q projected to 3D * Scale
                // Or use the "Moxness" shadow directly.

                const shadow = this.project4Dto3D(E8Lattice.foldTo4D(v8)); // Use un-normalized for position

                positions[idx * 3 + 0] = shadow[0] * scale;
                positions[idx * 3 + 1] = shadow[1] * scale;
                positions[idx * 3 + 2] = shadow[2] * scale;

                idx++;
            }
        }

        return { positions, rotations };
    }
}
