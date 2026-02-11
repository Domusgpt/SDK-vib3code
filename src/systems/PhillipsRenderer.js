/**
 * PhillipsRenderer.js
 * Core rendering system for the "Quatossian Inscription Framework".
 *
 * Features:
 * - Deterministic "Canonical View" rendering (Albedo only).
 * - "Cubit" Rendering: Points with intrinsic Quaternion Spin.
 * - Kirigami Modulation: Moiré interference patterns driven by spin phase.
 * - Uses Plastic Ratio for scale modulation.
 */

import { PLASTIC_CONSTANT } from '../math/Plastic.js';

export class PhillipsRenderer {
    /**
     * @param {HTMLCanvasElement} canvas - The canvas to render to.
     * @param {Object} options - Configuration options.
     */
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl', {
            alpha: true,
            antialias: false // Disable AA for sharper "pixel-perfect" canonical views
        });

        if (!this.gl) {
            throw new Error('PhillipsRenderer: WebGL not supported');
        }

        this.options = {
            plasticScale: PLASTIC_CONSTANT,
            backgroundColor: [0, 0, 0, 0], // Transparent by default
            kirigamiShift: [0, 0], // Moiré shift vector
            moireFreq: 20.0, // Frequency of the interference pattern
            ...options
        };

        this.program = null;
        this.buffers = {};
        this.uniforms = {};
        this.count = 0;

        this.init();
    }

    init() {
        this.initShaders();
        this.initBuffers();

        // Initial viewport setup
        this.resize();
    }

    initShaders() {
        // Vertex Shader
        const vertexSource = `
            attribute vec3 a_position;
            attribute float a_scale;
            attribute vec3 a_color;
            attribute vec4 a_rotation; // Quatossian Spin State (Quaternion)

            uniform mat4 u_viewProjection;
            uniform float u_plasticScale;
            uniform vec2 u_resolution;

            varying vec3 v_color;
            varying vec2 v_screenPos;
            varying vec4 v_rotation; // Pass spin to fragment

            void main() {
                // Project position
                vec4 pos = u_viewProjection * vec4(a_position, 1.0);
                gl_Position = pos;

                // Size attenuation
                float dist = pos.w;
                float size = (300.0 * a_scale * u_plasticScale) / dist;
                gl_PointSize = max(2.0, size);

                v_color = a_color;
                v_rotation = a_rotation;

                // Normalized screen position (-1 to 1)
                v_screenPos = pos.xy / pos.w;
            }
        `;

        // Fragment Shader: Quatossian Moiré Interference
        const fragmentSource = `
            precision mediump float;

            uniform vec2 u_kirigamiShift;
            uniform float u_moireFreq;

            varying vec3 v_color;
            varying vec2 v_screenPos;
            varying vec4 v_rotation;

            void main() {
                // Circular point shape
                vec2 coord = gl_PointCoord - vec2(0.5);
                float distSq = dot(coord, coord);
                if (distSq > 0.25) discard;

                // --- Kirigami Interference Logic ---

                // 1. Extract Phase from Quaternion Spin
                // We use the 'w' component (scalar) and a projection of the vector part
                // to determine the phase offset of this specific Cubit.
                // This makes every point's interference unique based on its 8D->4D fold.
                float spinPhase = v_rotation.w * 3.14159 + v_rotation.x;

                // 2. Moiré Pattern Generation
                // G1: Base Grid (Screen Space)
                float g1 = sin((v_screenPos.x + v_screenPos.y) * u_moireFreq + spinPhase);

                // G2: Shifted Grid (Kirigami Offset)
                // The shift creates the Moiré beats.
                float g2 = sin((v_screenPos.x + v_screenPos.y + u_kirigamiShift.x) * (u_moireFreq * 1.05) + spinPhase);

                // 3. Construct Interference Term
                // I = I0 * (1 + sin(Phase + Moiré))
                float interference = 0.5 + 0.5 * (g1 * g2);

                // 4. Modulate Intensity
                // "Diamond-Locked" coherence means stable, bright centers with shimmering edges.
                vec3 finalColor = v_color * (0.6 + 0.8 * interference);

                // Soft edge alpha
                float alpha = 1.0 - smoothstep(0.20, 0.25, distSq);

                gl_FragColor = vec4(finalColor, alpha);
            }
        `;

        this.program = this.createProgram(vertexSource, fragmentSource);

        if (this.program) {
            this.attributes = {
                position: this.gl.getAttribLocation(this.program, 'a_position'),
                scale: this.gl.getAttribLocation(this.program, 'a_scale'),
                color: this.gl.getAttribLocation(this.program, 'a_color'),
                rotation: this.gl.getAttribLocation(this.program, 'a_rotation')
            };

            this.uniforms = {
                viewProjection: this.gl.getUniformLocation(this.program, 'u_viewProjection'),
                plasticScale: this.gl.getUniformLocation(this.program, 'u_plasticScale'),
                resolution: this.gl.getUniformLocation(this.program, 'u_resolution'),
                kirigamiShift: this.gl.getUniformLocation(this.program, 'u_kirigamiShift'),
                moireFreq: this.gl.getUniformLocation(this.program, 'u_moireFreq')
            };
        }
    }

    createProgram(vertexSource, fragmentSource) {
        const gl = this.gl;
        const vertexShader = this.createShader(gl.VERTEX_SHADER, vertexSource);
        const fragmentShader = this.createShader(gl.FRAGMENT_SHADER, fragmentSource);

        if (!vertexShader || !fragmentShader) return null;

        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);

        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error('PhillipsRenderer Program Link Error:', gl.getProgramInfoLog(program));
            return null;
        }
        return program;
    }

    createShader(type, source) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('PhillipsRenderer Shader Compile Error:', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    initBuffers() {
        const gl = this.gl;
        this.buffers.position = gl.createBuffer();
        this.buffers.scale = gl.createBuffer();
        this.buffers.color = gl.createBuffer();
        this.buffers.rotation = gl.createBuffer();
    }

    /**
     * Populate the renderer with Quatossian Cubit data.
     * @param {Object} data
     * @param {Float32Array} data.positions - Flat [x,y,z...]
     * @param {Float32Array} data.scales - Flat [s...]
     * @param {Float32Array} data.colors - Flat [r,g,b...]
     * @param {Float32Array} data.rotations - Flat [x,y,z,w...] (Quaternions)
     */
    setData(data) {
        const gl = this.gl;

        if (data.positions) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
            gl.bufferData(gl.ARRAY_BUFFER, data.positions, gl.STATIC_DRAW);
            this.count = data.positions.length / 3;
        }

        if (data.scales) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.scale);
            gl.bufferData(gl.ARRAY_BUFFER, data.scales, gl.STATIC_DRAW);
        }

        if (data.colors) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.color);
            gl.bufferData(gl.ARRAY_BUFFER, data.colors, gl.STATIC_DRAW);
        }

        if (data.rotations) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.rotation);
            gl.bufferData(gl.ARRAY_BUFFER, data.rotations, gl.STATIC_DRAW);
        }
    }

    resize() {
        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight;

        if (this.canvas.width !== width || this.canvas.height !== height) {
            this.canvas.width = width;
            this.canvas.height = height;
            this.gl.viewport(0, 0, width, height);
        }
    }

    render(viewProjectionMatrix) {
        if (!this.program) return;

        const gl = this.gl;
        this.resize();

        // clear
        const bg = this.options.backgroundColor;
        gl.clearColor(bg[0], bg[1], bg[2], bg[3]);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        gl.useProgram(this.program);

        // Bind Attributes
        // Position
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
        gl.vertexAttribPointer(this.attributes.position, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.attributes.position);

        // Scale
        if (this.attributes.scale !== -1) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.scale);
            gl.vertexAttribPointer(this.attributes.scale, 1, gl.FLOAT, false, 0, 0);
            gl.enableVertexAttribArray(this.attributes.scale);
        }

        // Color
        if (this.attributes.color !== -1) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.color);
            gl.vertexAttribPointer(this.attributes.color, 3, gl.FLOAT, false, 0, 0);
            gl.enableVertexAttribArray(this.attributes.color);
        }

        // Rotation (Quaternion)
        if (this.attributes.rotation !== -1) {
            // If no rotation data provided, we need to disable the attrib or provide default
            // But setData assumes we have it.
            // Check if buffer is bound (size > 0).
            // Simplified: We assume data is provided if attrib is active.
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.rotation);
            gl.vertexAttribPointer(this.attributes.rotation, 4, gl.FLOAT, false, 0, 0);
            gl.enableVertexAttribArray(this.attributes.rotation);
        }

        // Set Uniforms
        gl.uniformMatrix4fv(this.uniforms.viewProjection, false, viewProjectionMatrix);
        gl.uniform1f(this.uniforms.plasticScale, this.options.plasticScale);
        gl.uniform2f(this.uniforms.resolution, this.canvas.width, this.canvas.height);

        const time = Date.now() * 0.001;
        const shiftX = this.options.kirigamiShift[0] || Math.sin(time * 0.5);
        const shiftY = this.options.kirigamiShift[1] || Math.cos(time * 0.5);
        gl.uniform2f(this.uniforms.kirigamiShift, shiftX, shiftY);
        gl.uniform1f(this.uniforms.moireFreq, this.options.moireFreq);

        // Draw
        gl.drawArrays(gl.POINTS, 0, this.count);
    }
}
