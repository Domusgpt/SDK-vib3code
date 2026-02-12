/**
 * QuatossianEngine.js
 * Adapter for the Phillips Rendering System within the VIB3+ Engine.
 *
 * Manages:
 * - E8 Lattice Generation (Quatossian Cloud)
 * - PhillipsRenderer instance
 * - Parameter mapping
 * - Animation loop
 */

import { PhillipsRenderer } from '../systems/PhillipsRenderer.js';
import { E8Lattice } from '../math/E8.js';
import { PLASTIC_CONSTANT } from '../math/Plastic.js';
import { Quaternion } from '../math/Quaternion.js';

export class QuatossianEngine {
    constructor(options = {}) {
        this.isActive = false;
        this.canvas = null;
        this.renderer = null;
        this.cloudData = null;
        this.autoStart = options.autoStart ?? true;

        // Parameters
        this.params = {
            shells: 20,
            moireFreq: 30.0,
            kirigamiShift: [0.1, 0.1],
            edgeSensitivity: 1.0,
            enable4D: true,
            speed: 1.0,
            plasticScale: PLASTIC_CONSTANT
        };

        this.time = 0;
    }

    /**
     * Initialize with a specific canvas.
     * @param {HTMLCanvasElement} canvas
     */
    init(canvas) {
        if (!canvas) {
            // Try to find default canvas if none provided
            canvas = document.getElementById('quatossian-content-canvas');
        }

        if (!canvas) {
            console.error("QuatossianEngine: No canvas found.");
            return false;
        }

        this.canvas = canvas;

        try {
            this.renderer = new PhillipsRenderer(this.canvas, {
                plasticScale: this.params.plasticScale,
                enable4D: this.params.enable4D,
                enablePostProcess: true
            });

            // Generate initial data
            this.updateGeometry();

            console.log("Quatossian Engine Initialized.");
            return true;
        } catch (e) {
            console.error("Quatossian Engine Init Failed:", e);
            return false;
        }
    }

    /**
     * Generates or Regenerates the Point Cloud based on parameters.
     */
    updateGeometry() {
        if (!this.renderer) return;

        const shells = Math.max(1, Math.min(50, this.params.shells));

        // Generate E8 Cloud
        const cloud = E8Lattice.generateCloud(shells);

        // Sort for spatial coherence (Z-Curve)
        // Optimization: Sort is expensive, do it only if needed or async.
        // For now, synchronous.
        const sortedCloud = E8Lattice.sortCloud(cloud);

        const count = sortedCloud.positions.length / 3;
        const scales = new Float32Array(count);
        const colors = new Float32Array(count * 3);

        for(let i=0; i<count; i++) {
            const shellIdx = Math.floor(i / 240);
            scales[i] = 1.0 / Math.pow(PLASTIC_CONSTANT, (shellIdx % 5) + 1);

            // Color from Spin State
            const r = sortedCloud.rotations;
            colors[i*3] = (r[i*4] + 1) * 0.5;
            colors[i*3+1] = (r[i*4+1] + 1) * 0.5;
            colors[i*3+2] = (r[i*4+2] + 1) * 0.5;
        }

        this.cloudData = {
            positions: sortedCloud.positions,
            rotations: sortedCloud.rotations,
            scales: scales,
            colors: colors
        };

        this.renderer.setData(this.cloudData);
    }

    /**
     * Update parameters from the engine.
     * @param {Object} newParams
     */
    updateParameters(newParams) {
        let geometryChanged = false;

        if (newParams.shells && newParams.shells !== this.params.shells) {
            this.params.shells = newParams.shells;
            geometryChanged = true;
        }

        if (newParams.gridDensity) {
            // Map gridDensity to shells (approx 10-100 -> 5-30)
            const shells = Math.max(5, Math.floor(newParams.gridDensity / 3));
            if (shells !== this.params.shells) {
                this.params.shells = shells;
                geometryChanged = true;
            }
        }

        if (newParams.speed !== undefined) this.params.speed = newParams.speed;
        if (newParams.moireFreq !== undefined) this.params.moireFreq = newParams.moireFreq;
        if (newParams.edgeSensitivity !== undefined) this.params.edgeSensitivity = newParams.edgeSensitivity;

        // Update Renderer Options
        if (this.renderer) {
            this.renderer.options.plasticScale = this.params.plasticScale;
            this.renderer.options.moireFreq = this.params.moireFreq;
            this.renderer.options.edgeSensitivity = this.params.edgeSensitivity;
        }

        if (geometryChanged) {
            this.updateGeometry();
        }
    }

    setActive(active) {
        this.isActive = active;
        // Show/Hide canvas container logic handled by CanvasManager usually,
        // but we can ensure our canvas is visible if needed.
    }

    renderFrame() {
        if (!this.isActive || !this.renderer) return;

        this.time += 0.01 * this.params.speed;

        // 4D Rotation Animation
        // Simple rotation in XW and YZ planes
        const angle = this.time;
        // q = [sin(t), 0, cos(t), 0]
        const qTime = [Math.sin(angle), 0, Math.cos(angle), 0];

        this.renderer.options.timeRotation = qTime;

        // Camera (Standard MVP for now)
        // Ideally this comes from a camera system
        const canvas = this.canvas;
        const aspect = canvas.width / canvas.height;
        const f = 1.0 / Math.tan(45 * Math.PI / 360);
        const proj = new Float32Array([
            f/aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, -1, -1,
            0, 0, -0.2, 0
        ]);

        // Model Matrix (Zoom out to see cloud)
        const model = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,-4,1];

        // MVP Multiply
        const mvp = new Float32Array(16);
        for(let i=0; i<4; i++) {
            for(let j=0; j<4; j++) {
                let sum = 0;
                for(let k=0; k<4; k++) sum += proj[k*4+j] * model[i*4+k];
                mvp[i*4+j] = sum;
            }
        }

        this.renderer.render(mvp);
    }

    destroy() {
        this.isActive = false;
        // Clean up GL context if needed, but usually CanvasManager kills the canvas.
        this.renderer = null;
        this.canvas = null;
    }
}
