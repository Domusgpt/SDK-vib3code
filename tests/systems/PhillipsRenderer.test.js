import { describe, it, expect, vi } from 'vitest';
import { PhillipsRenderer } from '../../src/systems/PhillipsRenderer.js';

// Mock WebGL
const mockGL = {
    getExtension: () => ({}),
    getParameter: () => 0,
    createShader: () => ({}),
    shaderSource: () => {},
    compileShader: () => {},
    getShaderParameter: () => true,
    getShaderInfoLog: () => "",
    createProgram: () => ({}),
    attachShader: () => {},
    linkProgram: () => {},
    getProgramParameter: () => true,
    getProgramInfoLog: () => "",
    getAttribLocation: () => 0,
    getUniformLocation: () => ({}),
    createBuffer: () => ({}),
    bindBuffer: () => {},
    bufferData: () => {},
    enableVertexAttribArray: () => {},
    vertexAttribPointer: () => {},
    useProgram: () => {},
    viewport: () => {},
    clearColor: () => {},
    clear: () => {},
    enable: () => {},
    blendFunc: () => {},
    uniformMatrix4fv: () => {},
    uniform1f: () => {},
    uniform2f: () => {},
    uniform4f: () => {},
    uniform1i: () => {},
    drawArrays: () => {},
    deleteBuffer: () => {},
    deleteProgram: () => {},
    deleteShader: () => {},

    // NEW MOCKS FOR FRAMEBUFFERS (v4)
    createFramebuffer: () => ({}),
    bindFramebuffer: () => {},
    deleteFramebuffer: () => {},
    createTexture: () => ({}),
    bindTexture: () => {},
    texImage2D: () => {},
    texParameteri: () => {},
    framebufferTexture2D: () => {},
    drawBuffers: () => {},
    activeTexture: () => {},
    deleteTexture: () => {},
    // Constants
    FRAMEBUFFER: 0x8D40,
    COLOR_ATTACHMENT0: 0x8CE0,
    TEXTURE_2D: 0x0DE1,
    RGBA: 0x1908,
    UNSIGNED_BYTE: 0x1401,
    FLOAT: 0x1406
};

describe('PhillipsRenderer', () => {

    it('should initialize with WebGL context', () => {
        const canvas = document.createElement('canvas');
        canvas.getContext = vi.fn().mockReturnValue(mockGL);

        const renderer = new PhillipsRenderer(canvas);
        expect(renderer).toBeDefined();
        expect(renderer.gl).toBe(mockGL);
        expect(renderer.program).toBeDefined();
        // Check framebuffers initialized
        expect(renderer.framebuffer).toBeDefined();
    });

    it('should support 4D mode', () => {
        const canvas = document.createElement('canvas');
        canvas.getContext = vi.fn().mockReturnValue(mockGL);

        const renderer = new PhillipsRenderer(canvas, { enable4D: true, enablePostProcess: false });
        expect(renderer.options.enable4D).toBe(true);

        const spy4f = vi.spyOn(mockGL, 'uniform4f');
        const vp = new Float32Array(16);
        renderer.render(vp);
        expect(spy4f).toHaveBeenCalled();
    });
});
