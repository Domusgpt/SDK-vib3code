/**
 * PhillipsRenderer.js
 * Core rendering system for the "Quatossian Inscription Framework".
 *
 * Features:
 * - Deterministic "Canonical View" rendering (Albedo only).
 * - "Cubit" Rendering: Points with intrinsic Quaternion Spin.
 * - GPU-Accelerated 4D Rotation (Moxness Mode).
 * - Multi-Pass Post-Processing: "Pancharatnam-Berry" Phase Edge Detection.
 */

import { PLASTIC_CONSTANT } from '../math/Plastic.js';

export class PhillipsRenderer {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl', {
            alpha: true,
            antialias: false,
            preserveDrawingBuffer: true
        });

        if (!this.gl) {
            throw new Error('PhillipsRenderer: WebGL not supported');
        }

        this.options = {
            plasticScale: PLASTIC_CONSTANT,
            backgroundColor: [0, 0, 0, 0],
            kirigamiShift: [0, 0],
            moireFreq: 20.0,
            enable4D: false,
            timeRotation: [0, 0, 0, 1],
            enablePostProcess: true, // New Option
            edgeSensitivity: 1.0, // Berry Flux control
            ...options
        };

        this.program = null;
        this.postProgram = null;
        this.framebuffer = null;
        this.textures = {}; // { color, spin }
        this.buffers = {};
        this.uniforms = {};
        this.count = 0;

        // Extensions for Float Textures (needed for Spin/Phase precision)
        this.gl.getExtension('OES_texture_float');
        this.gl.getExtension('WEBGL_draw_buffers'); // If implementing MRT, but we might do multipass via FBO swapping if MRT not available.
        // For WebGL1 compatibility, we might need multiple passes or pack data.
        // Let's assume we can pack Phase into Alpha or use a secondary pass?
        // Or just use a single pass that writes (Color.rgb, Phase.w) if Phase is scalar?
        // But Spin is Quaternion (4D).
        // We need MRT (Multiple Render Targets) or two passes.
        // WebGL1 MRT is rare (WEBGL_draw_buffers).
        // Let's assume we write Color + Phase(1D) to RGBA.
        // Or: Use WebGL2 if available?
        // The constructor creates 'webgl' (v1). Let's upgrade to 'webgl2' if possible, else fallback.

        this.isWebGL2 = typeof WebGL2RenderingContext !== 'undefined';
        if (this.isWebGL2) {
             const gl2 = canvas.getContext('webgl2', { alpha:true, antialias:false });
             if (gl2) {
                 this.gl = gl2;
                 this.gl.getExtension('EXT_color_buffer_float');
             }
        }

        this.init();
    }

    init() {
        this.initShaders();
        this.initPostShaders();
        this.initBuffers();
        this.initFramebuffers();
        this.resize();
    }

    initFramebuffers() {
        const gl = this.gl;
        const width = this.canvas.width;
        const height = this.canvas.height;

        if (this.framebuffer) gl.deleteFramebuffer(this.framebuffer);
        if (this.textures.color) gl.deleteTexture(this.textures.color);
        if (this.textures.spin) gl.deleteTexture(this.textures.spin);

        this.framebuffer = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);

        // Texture 1: Color (RGBA)
        this.textures.color = this.createTexture(width, height, gl.RGBA, gl.UNSIGNED_BYTE);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.textures.color, 0);

        // Texture 2: Spin (RGBA Float) - Stores Quaternion
        // Need WebGL2 or MRT extension for >1 attachment
        if (this.isWebGL2) {
            this.textures.spin = this.createTexture(width, height, gl.RGBA, gl.FLOAT);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.textures.spin, 0);
            gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
        } else {
            // Fallback for WebGL1: We only render Color for now, Post-Process will be limited.
            // Or we pack Phase into Alpha of Color.
            console.warn("PhillipsRenderer: WebGL2 required for Full Quatossian Post-Processing.");
        }

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    createTexture(w, h, format, type) {
        const gl = this.gl;
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, format, w, h, 0, format, type, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        return tex;
    }

    initShaders() {
        const is2 = this.isWebGL2;
        const ver = is2 ? '#version 300 es' : '';
        const att = is2 ? 'in' : 'attribute';
        const var_in = is2 ? 'in' : 'varying';
        const var_out = is2 ? 'out' : 'varying';

        // VERTEX SHADER
        const vs = `${ver}
            ${att} vec3 a_position;
            ${att} float a_scale;
            ${att} vec3 a_color;
            ${att} vec4 a_rotation;

            uniform mat4 u_viewProjection;
            uniform float u_plasticScale;
            uniform vec4 u_timeRotation;
            uniform bool u_enable4D;

            ${var_out} vec3 v_color;
            ${var_out} vec2 v_screenPos;
            ${var_out} vec4 v_rotation;

            vec4 quatMul(vec4 a, vec4 b) {
                return vec4(
                    a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                    a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                    a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
                );
            }

            void main() {
                vec3 finalPos = a_position;
                vec4 spinState = a_rotation;

                if (u_enable4D) {
                    vec4 pRot = quatMul(u_timeRotation, a_rotation);
                    float w = pRot.w;
                    float f = 1.0 / max(0.001, 2.0 - w);
                    finalPos = pRot.xyz * f;
                    spinState = pRot;
                }

                vec4 pos = u_viewProjection * vec4(finalPos, 1.0);
                gl_Position = pos;

                float size = (300.0 * a_scale * u_plasticScale) / pos.w;
                gl_PointSize = max(2.0, size);

                v_color = a_color;
                v_rotation = spinState;
                v_screenPos = pos.xy / pos.w;
            }
        `;

        // FRAGMENT SHADER (Writes to MRT)
        const outDecl = is2 ? `layout(location = 0) out vec4 fragColor; layout(location = 1) out vec4 fragSpin;` : `void main() {`;
        const fs = `${ver}
            precision mediump float;
            ${var_in} vec3 v_color;
            ${var_in} vec2 v_screenPos;
            ${var_in} vec4 v_rotation;

            ${outDecl.replace('void main() {', '')}

            void main() {
                vec2 coord = gl_PointCoord - vec2(0.5);
                if (dot(coord, coord) > 0.25) discard;

                ${is2 ?
                `fragColor = vec4(v_color, 1.0);
                 fragSpin = v_rotation;`
                :
                `gl_FragColor = vec4(v_color, 1.0);` // WebGL1 fallback
                }
            }
        `;

        this.program = this.createProgram(vs, fs);
        // ... (Attributes/Uniforms setup similar to before)
        const gl = this.gl;
        if(this.program) {
            this.attributes = {
                position: gl.getAttribLocation(this.program, 'a_position'),
                scale: gl.getAttribLocation(this.program, 'a_scale'),
                color: gl.getAttribLocation(this.program, 'a_color'),
                rotation: gl.getAttribLocation(this.program, 'a_rotation')
            };
            this.uniforms = {
                viewProjection: gl.getUniformLocation(this.program, 'u_viewProjection'),
                plasticScale: gl.getUniformLocation(this.program, 'u_plasticScale'),
                timeRotation: gl.getUniformLocation(this.program, 'u_timeRotation'),
                enable4D: gl.getUniformLocation(this.program, 'u_enable4D')
            };
        }
    }

    initPostShaders() {
        if (!this.isWebGL2) return;
        const gl = this.gl;

        // Fullscreen Quad Vertex Shader
        const vs = `#version 300 es
            in vec2 a_position;
            out vec2 v_uv;
            void main() {
                v_uv = a_position * 0.5 + 0.5;
                gl_Position = vec4(a_position, 0.0, 1.0);
            }
        `;

        // POST PROCESS: Pancharatnam-Berry Phase Detection
        const fs = `#version 300 es
            precision mediump float;
            uniform sampler2D u_texColor;
            uniform sampler2D u_texSpin;
            uniform vec2 u_resolution;
            uniform float u_edgeSensitivity;

            in vec2 v_uv;
            out vec4 finalColor;

            void main() {
                vec4 baseColor = texture(u_texColor, v_uv);
                vec4 centerSpin = texture(u_texSpin, v_uv);

                // Edge Detection by Phase Discontinuity
                // We sample neighbors and check quaternion distance
                float phaseDiff = 0.0;
                vec2 pixel = 1.0 / u_resolution;

                vec2 offsets[4] = vec2[](
                    vec2(1, 0), vec2(-1, 0), vec2(0, 1), vec2(0, -1)
                );

                for(int i=0; i<4; i++) {
                    vec4 neighborSpin = texture(u_texSpin, v_uv + offsets[i] * pixel);
                    // Quaternion dot product
                    float d = dot(centerSpin, neighborSpin);
                    // Geodesic distance (approx 1 - |dot|)
                    phaseDiff += (1.0 - abs(d));
                }

                // If phase difference is high, add Neon Glow
                float edge = smoothstep(0.01, 0.1, phaseDiff * u_edgeSensitivity);

                // Additive glow
                vec3 neon = vec3(0.0, 1.0, 1.0) * edge; // Cyan edge

                finalColor = vec4(baseColor.rgb + neon, 1.0);
            }
        `;

        this.postProgram = this.createProgram(vs, fs);
        this.quadBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
            -1, -1,  1, -1,  -1, 1,
            -1, 1,   1, -1,   1, 1
        ]), gl.STATIC_DRAW);
    }

    createProgram(vsSrc, fsSrc) {
        const gl = this.gl;
        const vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, vsSrc);
        gl.compileShader(vs);
        if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
            console.error("VS Error:", gl.getShaderInfoLog(vs));
            return null;
        }
        const fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, fsSrc);
        gl.compileShader(fs);
        if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
            console.error("FS Error:", gl.getShaderInfoLog(fs));
            return null;
        }
        const p = gl.createProgram();
        gl.attachShader(p, vs);
        gl.attachShader(p, fs);
        gl.linkProgram(p);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
            console.error("Link Error:", gl.getProgramInfoLog(p));
            return null;
        }
        return p;
    }

    // ... (initBuffers, setData, resize same as previous, just ensure they are preserved)
    initBuffers() { /* Same as before */
        const gl = this.gl;
        this.buffers.position = gl.createBuffer();
        this.buffers.scale = gl.createBuffer();
        this.buffers.color = gl.createBuffer();
        this.buffers.rotation = gl.createBuffer();
    }

    setData(data) { /* Same as before */
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
            this.initFramebuffers(); // Recreate textures on resize
        }
    }

    render(viewProjectionMatrix) {
        if (!this.program) return;
        const gl = this.gl;

        // 1. Render Scene to Framebuffer
        if (this.options.enablePostProcess && this.isWebGL2 && this.framebuffer) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
            gl.viewport(0, 0, this.canvas.width, this.canvas.height);
            gl.clearColor(0,0,0,0);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        } else {
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.viewport(0, 0, this.canvas.width, this.canvas.height);
            // ... clear ...
        }

        // ... (Drawing Logic from before) ...
        gl.useProgram(this.program);
        // Bind Attributes
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
        gl.vertexAttribPointer(this.attributes.position, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.attributes.position);
        // ... (Bind other attributes: scale, color, rotation)
        if (this.attributes.scale !== -1) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.scale);
            gl.vertexAttribPointer(this.attributes.scale, 1, gl.FLOAT, false, 0, 0);
            gl.enableVertexAttribArray(this.attributes.scale);
        }
        if (this.attributes.color !== -1) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.color);
            gl.vertexAttribPointer(this.attributes.color, 3, gl.FLOAT, false, 0, 0);
            gl.enableVertexAttribArray(this.attributes.color);
        }
        if (this.attributes.rotation !== -1) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.rotation);
            gl.vertexAttribPointer(this.attributes.rotation, 4, gl.FLOAT, false, 0, 0);
            gl.enableVertexAttribArray(this.attributes.rotation);
        }

        // Set Uniforms
        gl.uniformMatrix4fv(this.uniforms.viewProjection, false, viewProjectionMatrix);
        gl.uniform1f(this.uniforms.plasticScale, this.options.plasticScale);
        gl.uniform1i(this.uniforms.enable4D, this.options.enable4D ? 1 : 0);
        const q = this.options.timeRotation || [0,0,0,1];
        gl.uniform4f(this.uniforms.timeRotation, q[0], q[1], q[2], q[3]);

        gl.drawArrays(gl.POINTS, 0, this.count);

        // 2. Post-Process Pass
        if (this.options.enablePostProcess && this.isWebGL2 && this.framebuffer) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, null); // Back to screen
            gl.useProgram(this.postProgram);

            // Bind Textures
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, this.textures.color);
            gl.uniform1i(gl.getUniformLocation(this.postProgram, 'u_texColor'), 0);

            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, this.textures.spin);
            gl.uniform1i(gl.getUniformLocation(this.postProgram, 'u_texSpin'), 1);

            gl.uniform2f(gl.getUniformLocation(this.postProgram, 'u_resolution'), this.canvas.width, this.canvas.height);
            gl.uniform1f(gl.getUniformLocation(this.postProgram, 'u_edgeSensitivity'), this.options.edgeSensitivity);

            // Draw Quad
            gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
            const posLoc = gl.getAttribLocation(this.postProgram, 'a_position');
            gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
            gl.enableVertexAttribArray(posLoc);

            gl.drawArrays(gl.TRIANGLES, 0, 6);
        }
    }
}
