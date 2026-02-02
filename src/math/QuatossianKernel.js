/**
 * QuatossianKernel.js
 * The fundamental primitive of the framework.
 * A hybrid structure combining 3D Gaussian Splat properties with Quaternionic Spin states.
 */

import { Quaternion } from './Quaternion.js';
import { E8Lattice } from './E8.js';

export class QuatossianKernel {
    constructor() {
        this.position = new Float32Array(3); // x, y, z
        this.scale = new Float32Array(3);    // sx, sy, sz
        this.rotation = new Quaternion();    // qx, qy, qz, qw (Spin State)
        this.color = new Float32Array(3);    // r, g, b (Albedo)
        this.phase = 0.0;                    // Wave Phase (0-2PI)

        // High-Dimensional Index
        this.latticeIndex = 0;               // Morton/LSH key from E8 projection
    }

    /**
     * Inscribe this kernel into the E8 Lattice.
     * Updates the latticeIndex based on current position and phase.
     */
    inscribe() {
        // Project position + phase into 8D
        const v8 = E8Lattice.project3Dto8D(this.position[0], this.position[1], this.position[2]);

        // Use phase to modulate the 8th dimension ("Void")
        v8[7] = this.phase;

        // Quantize to nearest valid lattice site (snap to grid)
        const latticeNode = E8Lattice.quantizeToLattice(v8);

        // Generate Sort Key
        this.latticeIndex = E8Lattice.generateLSH(latticeNode);
    }

    /**
     * Set orientation from Euler angles (radians).
     */
    setRotation(x, y, z) {
        // Create temporary quaternions for axes and multiply
        // Simplified:
        const c1 = Math.cos(x/2), c2 = Math.cos(y/2), c3 = Math.cos(z/2);
        const s1 = Math.sin(x/2), s2 = Math.sin(y/2), s3 = Math.sin(z/2);

        this.rotation.x = s1 * c2 * c3 + c1 * s2 * s3;
        this.rotation.y = c1 * s2 * c3 - s1 * c2 * s3;
        this.rotation.z = c1 * c2 * s3 + s1 * s2 * c3;
        this.rotation.w = c1 * c2 * c3 - s1 * s2 * s3;
    }
}
