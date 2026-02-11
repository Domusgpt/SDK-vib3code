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
    drawArrays: () => {},
    deleteBuffer: () => {},
    deleteProgram: () => {},
    deleteShader: () => {}
};

describe('PhillipsRenderer', () => {

    it('should initialize with WebGL context', () => {
        const canvas = document.createElement('canvas');

        // Mock getContext
        canvas.getContext = vi.fn().mockReturnValue(mockGL);

        const renderer = new PhillipsRenderer(canvas);
        expect(renderer).toBeDefined();
        expect(renderer.gl).toBe(mockGL);
        expect(renderer.program).toBeDefined();
    });

    it('should set data correctly including rotations', () => {
        const canvas = document.createElement('canvas');
        canvas.getContext = vi.fn().mockReturnValue(mockGL);

        const renderer = new PhillipsRenderer(canvas);

        const data = {
            positions: new Float32Array([0,0,0]),
            scales: new Float32Array([1]),
            colors: new Float32Array([1,1,1]),
            rotations: new Float32Array([0,0,0,1])
        };

        // Spy on gl.bufferData
        // Need to spy on the mock object's method
        const spy = vi.spyOn(mockGL, 'bufferData');

        renderer.setData(data);

        // Should be called 4 times (pos, scale, col, rot)
        expect(spy).toHaveBeenCalledTimes(4);
    });
});
