/**
 * PhillipsRenderer.js
 * Core rendering system for the "Gaussian Flat" architecture.
 *
 * Features:
 * - Deterministic "Canonical View" rendering (Albedo only).
 * - No Spherical Harmonics (SH) or view-dependent effects.
 * - Uses Plastic Ratio for scale modulation.
 * - Extremely lightweight vertex/fragment shaders.
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
        // Vertex Shader: Projects 3D points to 2D screen space.
        // Uses a_scale driven by Plastic powers.
        const vertexSource = `
            attribute vec3 a_position;
            attribute float a_scale;
            attribute vec3 a_color; // Expecting normalized float 0-1 or we can use u8 normalized

            uniform mat4 u_viewProjection;
            uniform float u_plasticScale;
            uniform vec2 u_resolution;

            varying vec3 v_color;
            varying vec2 v_screenPos;

            void main() {
                // Project position
                vec4 pos = u_viewProjection * vec4(a_position, 1.0);
                gl_Position = pos;

                // Simple point size calculation based on depth and plastic scale
                // Perspective division for size attenuation
                float dist = pos.w;
                float size = (300.0 * a_scale * u_plasticScale) / dist;

                gl_PointSize = max(2.0, size); // Ensure minimum visibility

                v_color = a_color;

                // Pass normalized screen position for Moiré calculation
                v_screenPos = pos.xy / pos.w;
            }
        `;

        // Fragment Shader: The "Quatossian Flat Shader"
        // Now includes Kirigami Moiré Modulation for interference effects.
        const fragmentSource = `
            precision mediump float;

            uniform vec2 u_kirigamiShift;
            uniform float u_moireFreq;

            varying vec3 v_color;
            varying vec2 v_screenPos;

            void main() {
                // Circular point shape
                vec2 coord = gl_PointCoord - vec2(0.5);
                float distSq = dot(coord, coord);

                if (distSq > 0.25) {
                    discard;
                }

                // Kirigami-based Moiré Modulation
                // Simulating two overlapping grids (G1 and G2)
                // T(x) = Sum(G1(x) * G2(x + s))

                // Grid 1 (Base Mask)
                float g1 = sin((v_screenPos.x + v_screenPos.y) * u_moireFreq);

                // Grid 2 (Shifted Completer Mask)
                float g2 = sin((v_screenPos.x + v_screenPos.y + u_kirigamiShift.x) * (u_moireFreq * 1.05)); // Slight freq diff for beats

                // Interference Pattern (0.0 to 1.0)
                float interference = 0.5 + 0.5 * (g1 * g2);

                // Modulate opacity/intensity
                float alpha = 1.0 - smoothstep(0.24, 0.25, distSq);

                // "Shimmer" effect: Modulate color intensity by interference
                vec3 finalColor = v_color * (0.8 + 0.4 * interference);

                gl_FragColor = vec4(finalColor, alpha);
            }
        `;

        this.program = this.createProgram(vertexSource, fragmentSource);

        if (this.program) {
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
        // Create buffers but don't fill them yet.
        // setData() will handle that.
        const gl = this.gl;
        this.buffers.position = gl.createBuffer();
        this.buffers.scale = gl.createBuffer();
        this.buffers.color = gl.createBuffer();
    }

    /**
     * Populate the renderer with splat data.
     * @param {Object} data - Arrays of data.
     * @param {Float32Array} data.positions - Flat [x,y,z, x,y,z...]
     * @param {Float32Array} data.scales - Flat [s, s...]
     * @param {Float32Array} data.colors - Flat [r,g,b, r,g,b...] (Normalized 0-1)
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

    /**
     * Render the frame.
     * @param {Float32Array} viewProjectionMatrix - 4x4 View-Projection Matrix
     */
    render(viewProjectionMatrix) {
        if (!this.program) return;

        const gl = this.gl;
        this.resize();

        // clear
        const bg = this.options.backgroundColor;
        gl.clearColor(bg[0], bg[1], bg[2], bg[3]);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        // enable blending
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        gl.useProgram(this.program);

        // Bind attributes
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
        gl.enableVertexAttribArray(0); // a_position is index 0 usually, but better to look it up if not enforced
        // Wait, I didn't enforce locations in shader or look them up for attribs.
        // Let's do it dynamically.
        const aPosition = gl.getAttribLocation(this.program, 'a_position');
        const aScale = gl.getAttribLocation(this.program, 'a_scale');
        const aColor = gl.getAttribLocation(this.program, 'a_color');

        if (aPosition !== -1) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
            gl.vertexAttribPointer(aPosition, 3, gl.FLOAT, false, 0, 0);
            gl.enableVertexAttribArray(aPosition);
        }

        if (aScale !== -1) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.scale);
            gl.vertexAttribPointer(aScale, 1, gl.FLOAT, false, 0, 0);
            gl.enableVertexAttribArray(aScale);
        }

        if (aColor !== -1) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.color);
            gl.vertexAttribPointer(aColor, 3, gl.FLOAT, false, 0, 0);
            gl.enableVertexAttribArray(aColor);
        }

        // Set Uniforms
        gl.uniformMatrix4fv(this.uniforms.viewProjection, false, viewProjectionMatrix);
        gl.uniform1f(this.uniforms.plasticScale, this.options.plasticScale);
        gl.uniform2f(this.uniforms.resolution, this.canvas.width, this.canvas.height);

        // Dynamic Kirigami Uniforms (using Date.now for "shimmer" if not provided)
        const time = Date.now() * 0.001;
        const shiftX = this.options.kirigamiShift[0] || Math.sin(time);
        const shiftY = this.options.kirigamiShift[1] || Math.cos(time);
        gl.uniform2f(this.uniforms.kirigamiShift, shiftX, shiftY);
        gl.uniform1f(this.uniforms.moireFreq, this.options.moireFreq);

        // Draw
        gl.drawArrays(gl.POINTS, 0, this.count);
    }

    destroy() {
        const gl = this.gl;
        gl.deleteBuffer(this.buffers.position);
        gl.deleteBuffer(this.buffers.scale);
        gl.deleteBuffer(this.buffers.color);
        gl.deleteProgram(this.program);
    }
}
