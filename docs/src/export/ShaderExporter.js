/**
 * VIB3+ Shader Exporter
 *
 * Exports visualization parameters as shader code for various platforms:
 * - GLSL (WebGL/OpenGL)
 * - HLSL (Unreal Engine)
 * - ShaderGraph (Unity)
 *
 * @module ShaderExporter
 * @version 1.0.0
 */

export class ShaderExporter {
    constructor() {
        /**
         * Supported export formats
         */
        this.formats = {
            GLSL: 'glsl',
            HLSL: 'hlsl',
            UNITY_SHADERGRAPH: 'shadergraph',
            GODOT: 'gdshader'
        };

        /**
         * 6D rotation plane names
         */
        this.rotationPlanes = ['XY', 'XZ', 'YZ', 'XW', 'YW', 'ZW'];

        /**
         * 8 base geometry types
         */
        this.geometryTypes = [
            'Tetrahedron', 'Hypercube', 'Sphere', 'Torus',
            'KleinBottle', 'Fractal', 'Wave', 'Crystal'
        ];

        /**
         * 3 core types (polytope archetypes)
         */
        this.coreTypes = ['Base', 'HypersphereCore', 'HypertetrahedronCore'];
    }

    /**
     * Export current parameters as GLSL shader code
     * @param {Object} params - Current visualization parameters
     * @param {string} systemName - Active system (quantum/faceted/holographic)
     * @returns {string} Complete GLSL shader code
     */
    exportGLSL(params, systemName = 'faceted') {
        const header = this._generateHeader(params, systemName, 'GLSL');
        const uniforms = this._generateGLSLUniforms(params);
        const rotationFunctions = this._generateGLSLRotationFunctions();
        const geometryFunctions = this._generateGLSLGeometryFunctions();
        const mainFunction = this._generateGLSLMain(params);

        return `${header}

// ============================================================
// UNIFORMS - Parameter inputs
// ============================================================
${uniforms}

// ============================================================
// 6D ROTATION MATRICES
// ============================================================
${rotationFunctions}

// ============================================================
// 24 GEOMETRY FUNCTIONS (8 base × 3 core types)
// ============================================================
${geometryFunctions}

// ============================================================
// MAIN SHADER
// ============================================================
${mainFunction}
`;
    }

    /**
     * Export as HLSL for Unreal Engine
     * @param {Object} params - Current visualization parameters
     * @returns {string} HLSL material function code
     */
    exportHLSL(params) {
        const header = this._generateHeader(params, 'unreal', 'HLSL');

        return `${header}

// Unreal Engine Material Function
// Import as Custom Expression or Material Function

// Input Parameters (connect to Material inputs)
float Time;           // Use Time node
float2 UV;            // Use TexCoord node
float Geometry;       // ${params.geometry || 0} (0-23)
float6 Rotations;     // (${this._formatRotations(params)})
float GridDensity;    // ${params.gridDensity || 15}
float MorphFactor;    // ${params.morphFactor || 1.0}
float Chaos;          // ${params.chaos || 0.2}
float Hue;            // ${params.hue || 200}
float Intensity;      // ${params.intensity || 0.7}

// 4D Rotation Matrices
float4x4 RotateXW(float angle) {
    float c = cos(angle); float s = sin(angle);
    return float4x4(c,0,0,-s, 0,1,0,0, 0,0,1,0, s,0,0,c);
}

float4x4 RotateYW(float angle) {
    float c = cos(angle); float s = sin(angle);
    return float4x4(1,0,0,0, 0,c,0,-s, 0,0,1,0, 0,s,0,c);
}

float4x4 RotateZW(float angle) {
    float c = cos(angle); float s = sin(angle);
    return float4x4(1,0,0,0, 0,1,0,0, 0,0,c,-s, 0,0,s,c);
}

// Apply full 6D rotation
float4 Apply6DRotation(float4 pos, float6 rot) {
    // 3D rotations (XY, XZ, YZ) - standard
    pos = mul(RotateXY(rot.x), pos);
    pos = mul(RotateXZ(rot.y), pos);
    pos = mul(RotateYZ(rot.z), pos);
    // 4D rotations (XW, YW, ZW) - hyperspace
    pos = mul(RotateXW(rot.w), pos);
    pos = mul(RotateYW(rot[4]), pos);
    pos = mul(RotateZW(rot[5]), pos);
    return pos;
}

// Output: RGB color
return float3(${(params.hue || 200) / 360}, ${params.intensity || 0.7}, 1.0);
`;
    }

    /**
     * Export as Unity ShaderGraph Custom Function
     * @param {Object} params - Current visualization parameters
     * @returns {string} Unity Custom Function code
     */
    exportUnityShaderGraph(params) {
        return `// Unity ShaderGraph Custom Function
// Create: Right-click > Create Shader > Custom Function
// Paste this code in the Custom Function node

// VIB3+ 4D Geometry - Generated ${new Date().toISOString()}
// Parameters: Geometry=${params.geometry || 0}, Core=${Math.floor((params.geometry || 0) / 8)}

void VIB3_4DGeometry_float(
    float2 UV,
    float Time,
    float Geometry,
    float RotXW, float RotYW, float RotZW,
    float RotXY, float RotXZ, float RotYZ,
    float GridDensity,
    float MorphFactor,
    float Chaos,
    float Hue,
    float Intensity,
    out float3 Color,
    out float Alpha
) {
    // Create 4D point from UV
    float4 pos = float4(UV * 2.0 - 1.0, sin(Time * 0.3) * 0.5, cos(Time * 0.2) * 0.5);

    // Apply 6D rotation (simplified for ShaderGraph)
    // Full implementation requires matrix multiplication

    // HSV to RGB
    float h = Hue / 360.0;
    float3 rgb = saturate(abs(fmod(h * 6.0 + float3(0,4,2), 6.0) - 3.0) - 1.0);

    Color = lerp(float3(1,1,1), rgb, 0.8) * Intensity;
    Alpha = Intensity;
}
`;
    }

    /**
     * Export as Godot GDShader
     * @param {Object} params - Current visualization parameters
     * @returns {string} Godot shader code
     */
    exportGodot(params) {
        return `shader_type canvas_item;
// VIB3+ 4D Geometry Shader for Godot 4.x
// Generated: ${new Date().toISOString()}

// Parameters
uniform float geometry : hint_range(0, 23) = ${params.geometry || 0};
uniform float rot_xw : hint_range(0, 6.28) = ${params.rot4dXW || 0};
uniform float rot_yw : hint_range(0, 6.28) = ${params.rot4dYW || 0};
uniform float rot_zw : hint_range(0, 6.28) = ${params.rot4dZW || 0};
uniform float rot_xy : hint_range(0, 6.28) = ${params.rot4dXY || 0};
uniform float rot_xz : hint_range(0, 6.28) = ${params.rot4dXZ || 0};
uniform float rot_yz : hint_range(0, 6.28) = ${params.rot4dYZ || 0};
uniform float grid_density : hint_range(4, 50) = ${params.gridDensity || 15};
uniform float morph_factor : hint_range(0, 2) = ${params.morphFactor || 1};
uniform float chaos : hint_range(0, 1) = ${params.chaos || 0.2};
uniform float hue : hint_range(0, 360) = ${params.hue || 200};
uniform float intensity : hint_range(0, 1) = ${params.intensity || 0.7};

// 4D rotation matrix for XW plane
mat4 rotate_xw(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat4(
        vec4(c, 0, 0, -s),
        vec4(0, 1, 0, 0),
        vec4(0, 0, 1, 0),
        vec4(s, 0, 0, c)
    );
}

void fragment() {
    vec2 uv = (UV - 0.5) * 2.0;
    vec4 pos = vec4(uv / grid_density, sin(TIME * 0.3), cos(TIME * 0.2));

    // Apply morphing
    pos *= morph_factor;

    // HSV to RGB conversion
    float h = hue / 360.0;
    vec3 rgb = clamp(abs(mod(h * 6.0 + vec3(0, 4, 2), 6.0) - 3.0) - 1.0, 0.0, 1.0);

    COLOR = vec4(rgb * intensity, intensity);
}
`;
    }

    /**
     * Export parameters as JSON preset
     * @param {Object} params - Current parameters
     * @param {string} name - Preset name
     * @param {string} system - Active system
     * @returns {string} JSON string
     */
    exportPreset(params, name, system) {
        const preset = {
            name: name || 'Untitled Preset',
            version: '1.0.0',
            system: system || 'quantum',
            created: new Date().toISOString(),
            parameters: {
                geometry: params.geometry || 0,
                coreType: Math.floor((params.geometry || 0) / 8),
                baseGeometry: (params.geometry || 0) % 8,
                rotation: {
                    xy: params.rot4dXY || 0,
                    xz: params.rot4dXZ || 0,
                    yz: params.rot4dYZ || 0,
                    xw: params.rot4dXW || 0,
                    yw: params.rot4dYW || 0,
                    zw: params.rot4dZW || 0
                },
                visualization: {
                    gridDensity: params.gridDensity || 15,
                    morphFactor: params.morphFactor || 1.0,
                    chaos: params.chaos || 0.2,
                    speed: params.speed || 1.0
                },
                color: {
                    hue: params.hue || 200,
                    saturation: params.saturation || 0.8,
                    intensity: params.intensity || 0.7
                }
            },
            metadata: {
                geometryName: this.geometryTypes[(params.geometry || 0) % 8],
                coreTypeName: this.coreTypes[Math.floor((params.geometry || 0) / 8)]
            }
        };

        return JSON.stringify(preset, null, 2);
    }

    /**
     * Load preset from JSON
     * @param {string} json - JSON preset string
     * @returns {Object} Parsed parameters
     */
    loadPreset(json) {
        try {
            const preset = JSON.parse(json);
            return {
                geometry: preset.parameters.geometry,
                rot4dXY: preset.parameters.rotation.xy,
                rot4dXZ: preset.parameters.rotation.xz,
                rot4dYZ: preset.parameters.rotation.yz,
                rot4dXW: preset.parameters.rotation.xw,
                rot4dYW: preset.parameters.rotation.yw,
                rot4dZW: preset.parameters.rotation.zw,
                gridDensity: preset.parameters.visualization.gridDensity,
                morphFactor: preset.parameters.visualization.morphFactor,
                chaos: preset.parameters.visualization.chaos,
                speed: preset.parameters.visualization.speed,
                hue: preset.parameters.color.hue,
                saturation: preset.parameters.color.saturation,
                intensity: preset.parameters.color.intensity
            };
        } catch (e) {
            console.error('Failed to parse preset:', e);
            return null;
        }
    }

    // ============================================================
    // PRIVATE HELPER METHODS
    // ============================================================

    _generateHeader(params, system, format) {
        return `/**
 * VIB3+ 4D Procedural Shader
 * Format: ${format}
 * System: ${system}
 * Generated: ${new Date().toISOString()}
 *
 * Geometry: ${params.geometry || 0} (${this.geometryTypes[(params.geometry || 0) % 8]})
 * Core Type: ${this.coreTypes[Math.floor((params.geometry || 0) / 8)]}
 *
 * 6D Rotation: XY=${(params.rot4dXY || 0).toFixed(2)}, XZ=${(params.rot4dXZ || 0).toFixed(2)},
 *              YZ=${(params.rot4dYZ || 0).toFixed(2)}, XW=${(params.rot4dXW || 0).toFixed(2)},
 *              YW=${(params.rot4dYW || 0).toFixed(2)}, ZW=${(params.rot4dZW || 0).toFixed(2)}
 *
 * License: Royalty-free for commercial use
 * Generator: VIB3+ Engine v1.0
 */`;
    }

    _generateGLSLUniforms(params) {
        return `precision highp float;

uniform float u_time;
uniform vec2 u_resolution;

// Geometry (0-23: 8 base × 3 core types)
uniform float u_geometry;  // Current: ${params.geometry || 0}

// 6D Rotation
uniform float u_rot4dXY;   // Current: ${(params.rot4dXY || 0).toFixed(3)}
uniform float u_rot4dXZ;   // Current: ${(params.rot4dXZ || 0).toFixed(3)}
uniform float u_rot4dYZ;   // Current: ${(params.rot4dYZ || 0).toFixed(3)}
uniform float u_rot4dXW;   // Current: ${(params.rot4dXW || 0).toFixed(3)}
uniform float u_rot4dYW;   // Current: ${(params.rot4dYW || 0).toFixed(3)}
uniform float u_rot4dZW;   // Current: ${(params.rot4dZW || 0).toFixed(3)}

// Visualization
uniform float u_gridDensity;  // Current: ${params.gridDensity || 15}
uniform float u_morphFactor;  // Current: ${(params.morphFactor || 1).toFixed(2)}
uniform float u_chaos;        // Current: ${(params.chaos || 0.2).toFixed(2)}
uniform float u_hue;          // Current: ${params.hue || 200}
uniform float u_intensity;    // Current: ${(params.intensity || 0.7).toFixed(2)}`;
    }

    _generateGLSLRotationFunctions() {
        return `// 3D Rotation Matrices
mat4 rotateXY(float angle) {
    float c = cos(angle), s = sin(angle);
    return mat4(c,-s,0,0, s,c,0,0, 0,0,1,0, 0,0,0,1);
}

mat4 rotateXZ(float angle) {
    float c = cos(angle), s = sin(angle);
    return mat4(c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1);
}

mat4 rotateYZ(float angle) {
    float c = cos(angle), s = sin(angle);
    return mat4(1,0,0,0, 0,c,-s,0, 0,s,c,0, 0,0,0,1);
}

// 4D Rotation Matrices (Hyperspace)
mat4 rotateXW(float angle) {
    float c = cos(angle), s = sin(angle);
    return mat4(c,0,0,-s, 0,1,0,0, 0,0,1,0, s,0,0,c);
}

mat4 rotateYW(float angle) {
    float c = cos(angle), s = sin(angle);
    return mat4(1,0,0,0, 0,c,0,-s, 0,0,1,0, 0,s,0,c);
}

mat4 rotateZW(float angle) {
    float c = cos(angle), s = sin(angle);
    return mat4(1,0,0,0, 0,1,0,0, 0,0,c,-s, 0,0,s,c);
}

// Apply complete 6D rotation
vec4 apply6DRotation(vec4 pos) {
    pos = rotateXY(u_rot4dXY + u_time * 0.05) * pos;
    pos = rotateXZ(u_rot4dXZ + u_time * 0.06) * pos;
    pos = rotateYZ(u_rot4dYZ + u_time * 0.04) * pos;
    pos = rotateXW(u_rot4dXW + u_time * 0.07) * pos;
    pos = rotateYW(u_rot4dYW + u_time * 0.08) * pos;
    pos = rotateZW(u_rot4dZW + u_time * 0.09) * pos;
    return pos;
}`;
    }

    _generateGLSLGeometryFunctions() {
        return `// Base Geometry SDFs (0-7)
float baseGeometry(vec4 p, float type) {
    if (type < 0.5) {
        // 0: Tetrahedron
        return max(max(max(abs(p.x+p.y)-p.z, abs(p.x-p.y)-p.z),
                       abs(p.x+p.y)+p.z), abs(p.x-p.y)+p.z) / sqrt(3.0);
    } else if (type < 1.5) {
        // 1: Hypercube (Tesseract)
        vec4 q = abs(p) - 0.8;
        return length(max(q, 0.0)) + min(max(max(max(q.x,q.y),q.z),q.w), 0.0);
    } else if (type < 2.5) {
        // 2: Sphere
        return length(p) - 1.0;
    } else if (type < 3.5) {
        // 3: Torus
        vec2 t = vec2(length(p.xy) - 0.8, p.z);
        return length(t) - 0.3;
    } else if (type < 4.5) {
        // 4: Klein Bottle
        float r = length(p.xy);
        return abs(r - 0.7) - 0.2 + sin(atan(p.y,p.x)*3.0 + p.z*5.0) * 0.1;
    } else if (type < 5.5) {
        // 5: Fractal (Mandelbulb approx)
        return length(p) - 0.8 + sin(p.x*5.0)*sin(p.y*5.0)*sin(p.z*5.0) * 0.2;
    } else if (type < 6.5) {
        // 6: Wave
        return abs(p.z - sin(p.x*5.0+u_time)*cos(p.y*5.0+u_time)*0.3) - 0.1;
    } else {
        // 7: Crystal
        vec4 q = abs(p);
        return max(max(max(q.x,q.y),q.z),q.w) - 0.8;
    }
}

// Hypersphere Core wrapper (8-15)
float hypersphereCore(vec4 p, float baseType) {
    float baseShape = baseGeometry(p, baseType);
    float sphereField = length(p) - 1.2;
    return max(baseShape, sphereField);
}

// Hypertetrahedron Core wrapper (16-23)
float hypertetrahedronCore(vec4 p, float baseType) {
    float baseShape = baseGeometry(p, baseType);
    float tetraField = max(max(max(
        abs(p.x+p.y)-p.z-p.w,
        abs(p.x-p.y)-p.z+p.w),
        abs(p.x+p.y)+p.z-p.w),
        abs(p.x-p.y)+p.z+p.w) / sqrt(4.0);
    return max(baseShape, tetraField);
}

// Main geometry dispatcher (0-23)
float geometry(vec4 p, float type) {
    if (type < 8.0) {
        return baseGeometry(p, type);
    } else if (type < 16.0) {
        return hypersphereCore(p, type - 8.0);
    } else {
        return hypertetrahedronCore(p, type - 16.0);
    }
}`;
    }

    _generateGLSLMain(params) {
        return `void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / min(u_resolution.x, u_resolution.y);
    uv *= 2.0 / u_gridDensity;

    // Create 4D point
    vec4 pos = vec4(uv, sin(u_time * 0.3) * 0.5, cos(u_time * 0.2) * 0.5);

    // Apply full 6D rotation
    pos = apply6DRotation(pos);

    // Apply morphing and chaos
    pos *= u_morphFactor;
    pos += vec4(sin(u_time*0.1), cos(u_time*0.15), sin(u_time*0.12), cos(u_time*0.18)) * u_chaos;

    // Calculate distance to geometry
    float dist = geometry(pos, u_geometry);

    // Faceted rendering (sharp edges)
    float edge = smoothstep(0.02, 0.0, abs(dist));
    float fill = smoothstep(0.1, 0.0, dist) * 0.3;

    // HSV-based color from hue
    float hueVal = u_hue / 360.0 + dist * 0.2 + u_time * 0.05;
    vec3 color = vec3(
        0.5 + 0.5 * cos(hueVal * 6.28),
        0.5 + 0.5 * cos((hueVal + 0.33) * 6.28),
        0.5 + 0.5 * cos((hueVal + 0.67) * 6.28)
    );

    float alpha = (edge + fill) * u_intensity;
    gl_FragColor = vec4(color * alpha, alpha);
}`;
    }

    _formatRotations(params) {
        return `${(params.rot4dXY || 0).toFixed(2)}, ${(params.rot4dXZ || 0).toFixed(2)}, ${(params.rot4dYZ || 0).toFixed(2)}, ${(params.rot4dXW || 0).toFixed(2)}, ${(params.rot4dYW || 0).toFixed(2)}, ${(params.rot4dZW || 0).toFixed(2)}`;
    }
}

// Export singleton instance
export const shaderExporter = new ShaderExporter();
