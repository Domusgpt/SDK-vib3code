/**
 * VIB3+ MCP Server
 * Model Context Protocol server for agentic integration
 */

import { toolDefinitions, getToolList, validateToolInput } from './tools.js';
import { schemaRegistry } from '../../schemas/index.js';
import { telemetry, EventType, withTelemetry } from '../telemetry/index.js';

/**
 * Generate unique IDs
 */
function generateId(prefix = 'scene') {
    return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Geometry metadata lookup
 */
const BASE_GEOMETRIES = [
    { name: 'tetrahedron', vertices: 4, edges: 6, faces: 4 },
    { name: 'hypercube', vertices: 16, edges: 32, faces: 24 },
    { name: 'sphere', vertices: 'variable', edges: 'variable', faces: 'variable' },
    { name: 'torus', vertices: 'variable', edges: 'variable', faces: 'variable' },
    { name: 'klein_bottle', vertices: 'variable', edges: 'variable', faces: 'variable' },
    { name: 'fractal', vertices: 'variable', edges: 'variable', faces: 'variable' },
    { name: 'wave', vertices: 'variable', edges: 'variable', faces: 'variable' },
    { name: 'crystal', vertices: 8, edges: 12, faces: 6 }
];

const CORE_TYPES = ['base', 'hypersphere', 'hypertetrahedron'];

/**
 * MCP Server implementation
 */
export class MCPServer {
    constructor(engine = null) {
        this.engine = engine;
        this.sceneId = null;
        this.initialized = false;
    }

    /**
     * Set the VIB3 engine instance
     */
    setEngine(engine) {
        this.engine = engine;
        this.initialized = !!engine;
    }

    /**
     * Handle MCP tool call
     */
    async handleToolCall(toolName, args = {}) {
        const startTime = performance.now();

        // Validate input
        const validation = validateToolInput(toolName, args);
        if (!validation.valid) {
            telemetry.recordEvent(EventType.TOOL_INVOCATION_ERROR, {
                tool: toolName,
                error: validation.error.code
            });
            return { error: validation.error };
        }

        // Record start
        telemetry.recordEvent(EventType.TOOL_INVOCATION_START, { tool: toolName });

        try {
            let result;

            switch (toolName) {
                case 'create_4d_visualization':
                    result = await this.createVisualization(args);
                    break;
                case 'set_rotation':
                    result = this.setRotation(args);
                    break;
                case 'set_visual_parameters':
                    result = this.setVisualParameters(args);
                    break;
                case 'switch_system':
                    result = await this.switchSystem(args);
                    break;
                case 'change_geometry':
                    result = this.changeGeometry(args);
                    break;
                case 'get_state':
                    result = this.getState();
                    break;
                case 'randomize_parameters':
                    result = this.randomizeParameters(args);
                    break;
                case 'reset_parameters':
                    result = this.resetParameters();
                    break;
                case 'save_to_gallery':
                    result = this.saveToGallery(args);
                    break;
                case 'load_from_gallery':
                    result = this.loadFromGallery(args);
                    break;
                case 'search_geometries':
                    result = this.searchGeometries(args);
                    break;
                case 'get_parameter_schema':
                    result = this.getParameterSchema();
                    break;

                // === AGENTIC SAAS TOOLS ===
                case 'vib3_generate':
                    result = await this.generateFromPrompt(args);
                    break;
                case 'vib3_export_shader':
                    result = this.exportShader(args);
                    break;
                case 'vib3_batch':
                    result = await this.batchGenerate(args);
                    break;
                case 'render_preview':
                    result = await this.renderPreview(args);
                    break;
                case 'animate':
                    result = this.controlAnimation(args);
                    break;
                case 'get_capabilities':
                    result = this.getCapabilities(args);
                    break;
                default:
                    throw new Error(`Unknown tool: ${toolName}`);
            }

            // Record success
            const duration = performance.now() - startTime;
            telemetry.recordEvent(EventType.TOOL_INVOCATION_END, {
                tool: toolName,
                duration_ms: duration,
                success: true
            });

            return {
                success: true,
                operation: toolName,
                timestamp: new Date().toISOString(),
                duration_ms: duration,
                ...result
            };

        } catch (error) {
            const duration = performance.now() - startTime;
            telemetry.recordEvent(EventType.TOOL_INVOCATION_ERROR, {
                tool: toolName,
                duration_ms: duration,
                error: error.message
            });

            return {
                error: {
                    type: 'SystemError',
                    code: 'TOOL_EXECUTION_FAILED',
                    message: error.message,
                    suggestion: 'Check engine initialization and parameters',
                    retry_possible: true
                }
            };
        }
    }

    /**
     * Create a new visualization
     */
    async createVisualization(args) {
        const { system = 'quantum', geometry_index = 0, projection = 'perspective' } = args;

        if (this.engine) {
            await this.engine.switchSystem(system);
            if (geometry_index !== undefined) {
                this.engine.setParameter('geometry', geometry_index);
            }
        }

        this.sceneId = generateId('scene_4d');

        const baseIndex = geometry_index % 8;
        const coreIndex = Math.floor(geometry_index / 8);

        return {
            scene_id: this.sceneId,
            geometry: {
                index: geometry_index,
                base_type: BASE_GEOMETRIES[baseIndex].name,
                core_type: CORE_TYPES[coreIndex],
                vertex_count: BASE_GEOMETRIES[baseIndex].vertices,
                edge_count: BASE_GEOMETRIES[baseIndex].edges
            },
            system,
            projection,
            render_info: {
                complexity: geometry_index > 15 ? 'high' : geometry_index > 7 ? 'medium' : 'low',
                estimated_fps: 60
            },
            suggested_next_actions: ['set_rotation', 'set_visual_parameters', 'render_preview']
        };
    }

    /**
     * Set rotation parameters
     */
    setRotation(args) {
        if (this.engine) {
            if (args.XY !== undefined) this.engine.setParameter('rot4dXY', args.XY);
            if (args.XZ !== undefined) this.engine.setParameter('rot4dXZ', args.XZ);
            if (args.YZ !== undefined) this.engine.setParameter('rot4dYZ', args.YZ);
            if (args.XW !== undefined) this.engine.setParameter('rot4dXW', args.XW);
            if (args.YW !== undefined) this.engine.setParameter('rot4dYW', args.YW);
            if (args.ZW !== undefined) this.engine.setParameter('rot4dZW', args.ZW);

            telemetry.recordEvent(EventType.PARAMETER_CHANGE, { type: 'rotation' });
        }

        return {
            rotation_state: {
                XY: args.XY ?? 0,
                XZ: args.XZ ?? 0,
                YZ: args.YZ ?? 0,
                XW: args.XW ?? 0,
                YW: args.YW ?? 0,
                ZW: args.ZW ?? 0
            },
            suggested_next_actions: ['set_visual_parameters', 'render_preview']
        };
    }

    /**
     * Set visual parameters
     */
    setVisualParameters(args) {
        if (this.engine) {
            for (const [key, value] of Object.entries(args)) {
                this.engine.setParameter(key, value);
            }
            telemetry.recordEvent(EventType.PARAMETER_BATCH_CHANGE, {
                count: Object.keys(args).length
            });
        }

        return {
            parameters_updated: Object.keys(args),
            suggested_next_actions: ['render_preview', 'save_to_gallery']
        };
    }

    /**
     * Switch visualization system
     */
    async switchSystem(args) {
        const { system } = args;

        if (this.engine) {
            await this.engine.switchSystem(system);
            telemetry.recordEvent(EventType.SYSTEM_SWITCH, { system });
        }

        return {
            active_system: system,
            available_geometries: 24,
            suggested_next_actions: ['change_geometry', 'set_visual_parameters']
        };
    }

    /**
     * Change geometry
     */
    changeGeometry(args) {
        let geometryIndex = args.geometry_index;

        // Convert from base_type + core_type if provided
        if (args.base_type !== undefined) {
            const baseIndex = BASE_GEOMETRIES.findIndex(g => g.name === args.base_type);
            if (baseIndex === -1) {
                return {
                    error: {
                        type: 'ValidationError',
                        code: 'INVALID_BASE_TYPE',
                        message: `Unknown base type: ${args.base_type}`,
                        valid_options: BASE_GEOMETRIES.map(g => g.name),
                        suggestion: 'Use one of the valid base geometry types'
                    }
                };
            }

            const coreIndex = CORE_TYPES.indexOf(args.core_type || 'base');
            geometryIndex = coreIndex * 8 + baseIndex;
        }

        if (this.engine && geometryIndex !== undefined) {
            this.engine.setParameter('geometry', geometryIndex);
            telemetry.recordEvent(EventType.GEOMETRY_CHANGE, { geometry: geometryIndex });
        }

        const baseIndex = geometryIndex % 8;
        const coreIndex = Math.floor(geometryIndex / 8);

        return {
            geometry: {
                index: geometryIndex,
                base_type: BASE_GEOMETRIES[baseIndex].name,
                core_type: CORE_TYPES[coreIndex]
            },
            suggested_next_actions: ['set_rotation', 'set_visual_parameters']
        };
    }

    /**
     * Get current state
     */
    getState() {
        let params = {};
        let system = 'quantum';

        if (this.engine) {
            params = this.engine.getAllParameters();
            system = this.engine.getCurrentSystem();
        }

        const geometryIndex = params.geometry || 0;
        const baseIndex = geometryIndex % 8;
        const coreIndex = Math.floor(geometryIndex / 8);

        return {
            scene_id: this.sceneId,
            system,
            geometry: {
                index: geometryIndex,
                base_type: BASE_GEOMETRIES[baseIndex].name,
                core_type: CORE_TYPES[coreIndex]
            },
            rotation_state: {
                XY: params.rot4dXY || 0,
                XZ: params.rot4dXZ || 0,
                YZ: params.rot4dYZ || 0,
                XW: params.rot4dXW || 0,
                YW: params.rot4dYW || 0,
                ZW: params.rot4dZW || 0
            },
            visual: {
                dimension: params.dimension,
                gridDensity: params.gridDensity,
                morphFactor: params.morphFactor,
                chaos: params.chaos,
                speed: params.speed,
                hue: params.hue,
                intensity: params.intensity,
                saturation: params.saturation
            },
            suggested_next_actions: ['set_rotation', 'set_visual_parameters', 'save_to_gallery']
        };
    }

    /**
     * Randomize parameters
     */
    randomizeParameters(args) {
        if (this.engine) {
            this.engine.randomizeAll();
            telemetry.recordEvent(EventType.PARAMETER_RANDOMIZE, {});
        }

        return {
            ...this.getState(),
            randomized: true
        };
    }

    /**
     * Reset parameters
     */
    resetParameters() {
        if (this.engine) {
            this.engine.resetAll();
            telemetry.recordEvent(EventType.PARAMETER_RESET, {});
        }

        return {
            ...this.getState(),
            reset: true
        };
    }

    /**
     * Save to gallery
     */
    saveToGallery(args) {
        const { slot, name } = args;

        telemetry.recordEvent(EventType.GALLERY_SAVE, { slot });

        return {
            slot,
            name: name || `Variation ${slot}`,
            saved_at: new Date().toISOString(),
            suggested_next_actions: ['load_from_gallery', 'randomize_parameters']
        };
    }

    /**
     * Load from gallery
     */
    loadFromGallery(args) {
        const { slot } = args;

        if (this.engine) {
            // Apply variation
            const params = this.engine.parameters?.generateVariationParameters?.(slot) || {};
            this.engine.setParameters(params);
        }

        telemetry.recordEvent(EventType.GALLERY_LOAD, { slot });

        return {
            slot,
            loaded_at: new Date().toISOString(),
            ...this.getState()
        };
    }

    /**
     * Search geometries
     */
    searchGeometries(args) {
        const { core_type = 'all' } = args;

        const geometries = [];
        for (let i = 0; i < 24; i++) {
            const baseIndex = i % 8;
            const coreIndex = Math.floor(i / 8);
            const coreTypeName = CORE_TYPES[coreIndex];

            if (core_type !== 'all' && coreTypeName !== core_type) continue;

            geometries.push({
                index: i,
                base_type: BASE_GEOMETRIES[baseIndex].name,
                core_type: coreTypeName,
                vertex_count: BASE_GEOMETRIES[baseIndex].vertices,
                edge_count: BASE_GEOMETRIES[baseIndex].edges,
                description: this.getGeometryDescription(baseIndex, coreIndex)
            });
        }

        return {
            count: geometries.length,
            geometries,
            encoding_formula: 'geometry_index = core_index * 8 + base_index'
        };
    }

    /**
     * Get geometry description
     */
    getGeometryDescription(baseIndex, coreIndex) {
        const baseDesc = {
            0: 'Simple 4-vertex lattice, fundamental polytope',
            1: '4D cube projection with 16 vertices and 32 edges',
            2: 'Radial harmonic sphere with smooth surfaces',
            3: 'Toroidal field with continuous surface',
            4: 'Non-orientable surface with topological twist',
            5: 'Recursive subdivision with self-similar structure',
            6: 'Sinusoidal interference patterns',
            7: 'Octahedral crystal structure'
        };

        const coreDesc = {
            0: 'Pure base geometry',
            1: 'Wrapped in 4D hypersphere using warpHypersphereCore()',
            2: 'Wrapped in 4D hypertetrahedron using warpHypertetraCore()'
        };

        return `${baseDesc[baseIndex]}. ${coreDesc[coreIndex]}`;
    }

    /**
     * Get parameter schema
     */
    getParameterSchema() {
        return {
            schema: schemaRegistry.getSchema('parameters'),
            usage: 'Validate parameters using this schema before setting them',
            suggested_next_actions: ['set_rotation', 'set_visual_parameters']
        };
    }

    /**
     * Get available tools (for progressive disclosure)
     */
    listTools(includeSchemas = false) {
        if (includeSchemas) {
            return getToolList();
        }

        return Object.keys(toolDefinitions).map(name => ({
            name,
            description: toolDefinitions[name].description
        }));
    }

    // ============================================================
    // AGENTIC SAAS TOOL IMPLEMENTATIONS
    // ============================================================

    /**
     * Natural language to visualization parameters mapping
     */
    promptToParams(prompt) {
        const lowerPrompt = prompt.toLowerCase();

        // Extract mood/style
        let system = 'quantum';
        if (lowerPrompt.includes('sharp') || lowerPrompt.includes('geometric') || lowerPrompt.includes('clean') || lowerPrompt.includes('professional')) {
            system = 'faceted';
        } else if (lowerPrompt.includes('layered') || lowerPrompt.includes('ethereal') || lowerPrompt.includes('holographic') || lowerPrompt.includes('rainbow')) {
            system = 'holographic';
        } else if (lowerPrompt.includes('organic') || lowerPrompt.includes('flowing') || lowerPrompt.includes('quantum') || lowerPrompt.includes('soft')) {
            system = 'quantum';
        }

        // Extract color/hue
        let hue = 200; // Default blue
        if (lowerPrompt.includes('red') || lowerPrompt.includes('warm') || lowerPrompt.includes('fire') || lowerPrompt.includes('energetic')) {
            hue = 0 + Math.random() * 30;
        } else if (lowerPrompt.includes('orange') || lowerPrompt.includes('sunset')) {
            hue = 30;
        } else if (lowerPrompt.includes('yellow') || lowerPrompt.includes('gold') || lowerPrompt.includes('bright')) {
            hue = 50;
        } else if (lowerPrompt.includes('green') || lowerPrompt.includes('nature') || lowerPrompt.includes('forest')) {
            hue = 120;
        } else if (lowerPrompt.includes('cyan') || lowerPrompt.includes('aqua') || lowerPrompt.includes('ocean')) {
            hue = 180;
        } else if (lowerPrompt.includes('blue') || lowerPrompt.includes('calm') || lowerPrompt.includes('cool')) {
            hue = 220;
        } else if (lowerPrompt.includes('purple') || lowerPrompt.includes('violet') || lowerPrompt.includes('mystic')) {
            hue = 280;
        } else if (lowerPrompt.includes('pink') || lowerPrompt.includes('magenta')) {
            hue = 320;
        }

        // Extract geometry
        let geometry = 2; // Default sphere
        if (lowerPrompt.includes('cube') || lowerPrompt.includes('hypercube') || lowerPrompt.includes('tesseract')) {
            geometry = 1;
        } else if (lowerPrompt.includes('tetrahedron') || lowerPrompt.includes('pyramid')) {
            geometry = 0;
        } else if (lowerPrompt.includes('sphere') || lowerPrompt.includes('ball') || lowerPrompt.includes('orb')) {
            geometry = 2;
        } else if (lowerPrompt.includes('torus') || lowerPrompt.includes('donut') || lowerPrompt.includes('ring')) {
            geometry = 3;
        } else if (lowerPrompt.includes('klein') || lowerPrompt.includes('twisted')) {
            geometry = 4;
        } else if (lowerPrompt.includes('fractal') || lowerPrompt.includes('recursive') || lowerPrompt.includes('complex')) {
            geometry = 5;
        } else if (lowerPrompt.includes('wave') || lowerPrompt.includes('ripple') || lowerPrompt.includes('fluid')) {
            geometry = 6;
        } else if (lowerPrompt.includes('crystal') || lowerPrompt.includes('gem') || lowerPrompt.includes('diamond')) {
            geometry = 7;
        }

        // Apply core type modifiers
        if (lowerPrompt.includes('hyper') || lowerPrompt.includes('4d') || lowerPrompt.includes('dimensional')) {
            geometry += 8; // Hypersphere core
        }
        if (lowerPrompt.includes('tetra') || lowerPrompt.includes('angular')) {
            geometry = (geometry % 8) + 16; // Hypertetrahedron core
        }

        // Extract intensity/energy
        let intensity = 0.5;
        let chaos = 0.2;
        let speed = 1.0;
        if (lowerPrompt.includes('energetic') || lowerPrompt.includes('dynamic') || lowerPrompt.includes('intense') || lowerPrompt.includes('explosion')) {
            intensity = 0.8 + Math.random() * 0.2;
            chaos = 0.5 + Math.random() * 0.3;
            speed = 1.5 + Math.random() * 1.0;
        } else if (lowerPrompt.includes('calm') || lowerPrompt.includes('gentle') || lowerPrompt.includes('peaceful') || lowerPrompt.includes('serene')) {
            intensity = 0.3 + Math.random() * 0.2;
            chaos = 0.1;
            speed = 0.3 + Math.random() * 0.3;
        } else if (lowerPrompt.includes('psychedelic') || lowerPrompt.includes('trippy') || lowerPrompt.includes('wild')) {
            intensity = 1.0;
            chaos = 0.7 + Math.random() * 0.3;
            speed = 2.0 + Math.random() * 1.0;
        }

        // Extract saturation
        let saturation = 0.8;
        if (lowerPrompt.includes('muted') || lowerPrompt.includes('desaturated') || lowerPrompt.includes('gray') || lowerPrompt.includes('professional')) {
            saturation = 0.2 + Math.random() * 0.2;
        } else if (lowerPrompt.includes('vibrant') || lowerPrompt.includes('saturated') || lowerPrompt.includes('bold')) {
            saturation = 0.9 + Math.random() * 0.1;
        }

        return {
            system,
            geometry: Math.min(23, Math.max(0, geometry)),
            hue: hue % 360,
            saturation,
            intensity,
            chaos,
            speed,
            morphFactor: lowerPrompt.includes('morph') ? 1.5 : 1.0,
            gridDensity: lowerPrompt.includes('detailed') ? 30 : lowerPrompt.includes('simple') ? 8 : 15
        };
    }

    /**
     * Generate visualization from natural language prompt
     */
    async generateFromPrompt(args) {
        const { prompt, output_format = 'png', resolution = { width: 1024, height: 1024 } } = args;

        // Map prompt to parameters
        const params = this.promptToParams(prompt);

        // Apply to engine
        if (this.engine) {
            await this.engine.switchSystem(params.system);
            for (const [key, value] of Object.entries(params)) {
                if (key !== 'system') {
                    this.engine.setParameter(key, value);
                }
            }
        }

        telemetry.recordEvent(EventType.TOOL_INVOCATION_START, {
            tool: 'vib3_generate',
            prompt_length: prompt.length
        });

        // Generate preview
        const preview = await this.renderPreview({
            format: output_format === 'json' ? 'base64' : output_format,
            width: resolution.width,
            height: resolution.height,
            include_metadata: true
        });

        return {
            interpreted_as: params,
            original_prompt: prompt,
            ...preview,
            suggested_next_actions: ['set_rotation', 'vib3_export_shader', 'vib3_batch']
        };
    }

    /**
     * Export visualization as shader code
     */
    exportShader(args) {
        const { target, parameters, include_uniforms = true, optimize = false } = args;

        // Get current parameters if not provided
        let params = parameters;
        if (!params && this.engine) {
            params = this.engine.getAllParameters();
        }
        params = params || {};

        // Generate shader based on target
        const shader = this.generateShaderCode(target, params, include_uniforms, optimize);

        telemetry.recordEvent(EventType.TOOL_INVOCATION_END, {
            tool: 'vib3_export_shader',
            target,
            lines: shader.code.split('\n').length
        });

        return {
            target,
            format: this.getShaderFormat(target),
            code: shader.code,
            uniforms: shader.uniforms,
            instructions: shader.instructions,
            lines: shader.code.split('\n').length,
            suggested_next_actions: ['render_preview', 'vib3_batch']
        };
    }

    /**
     * Generate shader code for different platforms
     */
    generateShaderCode(target, params, includeUniforms, optimize) {
        const hue = (params.hue || 200) / 360.0;
        const intensity = params.intensity || 0.5;
        const chaos = params.chaos || 0.2;

        let code = '';
        let uniforms = [];
        let instructions = '';

        switch (target) {
            case 'glsl':
                uniforms = ['u_time', 'u_hue', 'u_intensity', 'u_chaos', 'u_rotation'];
                code = this.generateGLSL(params, includeUniforms);
                instructions = 'Copy this fragment shader to your WebGL/OpenGL project. Provide uniform values for animation.';
                break;

            case 'hlsl_unreal':
                uniforms = ['Time', 'Hue', 'Intensity', 'Chaos', 'Rotation'];
                code = this.generateHLSL(params, includeUniforms);
                instructions = 'Create a Material in Unreal Engine, add a Custom node, and paste this code. Connect outputs to material channels.';
                break;

            case 'unity_shadergraph':
                code = this.generateUnityShaderGraph(params);
                instructions = 'Import this as a .shadergraph file in Unity. Open in Shader Graph editor to customize.';
                break;

            case 'godot':
                uniforms = ['time', 'hue', 'intensity', 'chaos'];
                code = this.generateGodotShader(params, includeUniforms);
                instructions = 'Create a new ShaderMaterial in Godot and paste this code. Attach to a MeshInstance or ColorRect.';
                break;
        }

        return { code, uniforms, instructions };
    }

    /**
     * Generate GLSL shader code
     */
    generateGLSL(params, includeUniforms) {
        const hue = ((params.hue || 200) / 360.0).toFixed(4);
        const intensity = (params.intensity || 0.5).toFixed(4);
        const chaos = (params.chaos || 0.2).toFixed(4);
        const geometry = params.geometry || 2;

        return `// VIB3+ Generated GLSL Shader
// Geometry: ${BASE_GEOMETRIES[geometry % 8].name} with ${CORE_TYPES[Math.floor(geometry / 8)]} core
precision highp float;

${includeUniforms ? `uniform float u_time;
uniform float u_hue;
uniform float u_intensity;
uniform float u_chaos;
uniform mat4 u_rotation;
uniform vec2 u_resolution;` : `// Baked parameters
#define u_hue ${hue}
#define u_intensity ${intensity}
#define u_chaos ${chaos}
#define u_time 0.0`}

// 4D rotation matrix (6 planes)
mat4 rotate4D(float xy, float xz, float yz, float xw, float yw, float zw) {
    mat4 r = mat4(1.0);
    // XY rotation
    float c = cos(xy), s = sin(xy);
    r[0][0] = c; r[0][1] = -s; r[1][0] = s; r[1][1] = c;
    // Additional rotations would be composed here
    return r;
}

// HSV to RGB conversion
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// SDF for 4D geometry projection
float sdf4D(vec3 p, float w) {
    vec4 p4 = vec4(p, w);
    // Apply 4D rotation
    p4 = ${includeUniforms ? 'u_rotation *' : ''} p4;

    // Base geometry SDF (sphere example)
    float d = length(p4) - 1.0;

    // Add chaos distortion
    d += sin(p4.x * 10.0 + u_time) * sin(p4.y * 10.0) * sin(p4.z * 10.0) * u_chaos * 0.1;

    return d;
}

void main() {
    vec2 uv = (gl_FragCoord.xy * 2.0 - ${includeUniforms ? 'u_resolution' : 'vec2(1024.0)'}) / min(${includeUniforms ? 'u_resolution.x, u_resolution.y' : '1024.0, 1024.0'});

    // Ray direction
    vec3 rd = normalize(vec3(uv, 1.0));
    vec3 ro = vec3(0.0, 0.0, -3.0);

    // Ray march
    float t = 0.0;
    float w = sin(u_time * 0.5) * 0.5; // 4D slice position

    for (int i = 0; i < 64; i++) {
        vec3 p = ro + rd * t;
        float d = sdf4D(p, w);
        if (d < 0.001) break;
        t += d;
        if (t > 20.0) break;
    }

    // Coloring
    vec3 col = vec3(0.0);
    if (t < 20.0) {
        float hueShift = u_hue + t * 0.1;
        col = hsv2rgb(vec3(hueShift, 0.8, u_intensity));
    }

    gl_FragColor = vec4(col, 1.0);
}`;
    }

    /**
     * Generate HLSL shader code for Unreal
     */
    generateHLSL(params, includeUniforms) {
        const hue = ((params.hue || 200) / 360.0).toFixed(4);
        const intensity = (params.intensity || 0.5).toFixed(4);

        return `// VIB3+ Generated HLSL for Unreal Engine
// Material Custom Expression

float3 HSVtoRGB(float3 HSV) {
    float4 K = float4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    float3 P = abs(frac(HSV.xxx + K.xyz) * 6.0 - K.www);
    return HSV.z * lerp(K.xxx, saturate(P - K.xxx), HSV.y);
}

float SDF4D(float3 p, float w, float Time) {
    float4 p4 = float4(p, w);
    float d = length(p4) - 1.0;
    d += sin(p4.x * 10.0 + Time) * sin(p4.y * 10.0) * 0.1;
    return d;
}

// Main function - connect to Emissive Color
float3 result;
float2 uv = (Parameters.TexCoords[0].xy - 0.5) * 2.0;
float3 rd = normalize(float3(uv, 1.0));
float3 ro = float3(0.0, 0.0, -3.0);

float t = 0.0;
float w = sin(Time * 0.5) * 0.5;

[unroll]
for (int i = 0; i < 64; i++) {
    float3 p = ro + rd * t;
    float d = SDF4D(p, w, Time);
    if (d < 0.001) break;
    t += d;
    if (t > 20.0) break;
}

if (t < 20.0) {
    float hue = ${hue} + t * 0.1;
    result = HSVtoRGB(float3(hue, 0.8, ${intensity}));
} else {
    result = float3(0.0, 0.0, 0.0);
}

return result;`;
    }

    /**
     * Generate Unity ShaderGraph JSON
     */
    generateUnityShaderGraph(params) {
        return `{
  "m_ObjectId": "vib3_generated_shader",
  "m_Name": "VIB3_4D_Visualization",
  "m_Type": "UnlitShader",
  "m_Properties": {
    "Hue": { "type": "Float", "value": ${(params.hue || 200) / 360} },
    "Intensity": { "type": "Float", "value": ${params.intensity || 0.5} },
    "AnimationSpeed": { "type": "Float", "value": ${params.speed || 1.0} }
  },
  "m_SubGraphOutputs": ["BaseColor", "Alpha"],
  "m_Instructions": "Import as .shadergraph asset in Unity. Uses Custom Function nodes for 4D math."
}`;
    }

    /**
     * Generate Godot GDShader code
     */
    generateGodotShader(params, includeUniforms) {
        const hue = ((params.hue || 200) / 360.0).toFixed(4);
        const intensity = (params.intensity || 0.5).toFixed(4);

        return `shader_type canvas_item;
// VIB3+ Generated Godot Shader

${includeUniforms ? `uniform float hue : hint_range(0.0, 1.0) = ${hue};
uniform float intensity : hint_range(0.0, 1.0) = ${intensity};
uniform float animation_speed : hint_range(0.1, 3.0) = 1.0;` : ''}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float sdf_4d(vec3 p, float w) {
    vec4 p4 = vec4(p, w);
    float d = length(p4) - 1.0;
    d += sin(p4.x * 10.0 + TIME) * sin(p4.y * 10.0) * 0.1;
    return d;
}

void fragment() {
    vec2 uv = (UV - 0.5) * 2.0;
    vec3 rd = normalize(vec3(uv, 1.0));
    vec3 ro = vec3(0.0, 0.0, -3.0);

    float t = 0.0;
    float w = sin(TIME * ${includeUniforms ? 'animation_speed' : '1.0'} * 0.5) * 0.5;

    for (int i = 0; i < 64; i++) {
        vec3 p = ro + rd * t;
        float d = sdf_4d(p, w);
        if (d < 0.001) break;
        t += d;
        if (t > 20.0) break;
    }

    vec3 col = vec3(0.0);
    if (t < 20.0) {
        float h = ${includeUniforms ? 'hue' : hue} + t * 0.1;
        col = hsv2rgb(vec3(h, 0.8, ${includeUniforms ? 'intensity' : intensity}));
    }

    COLOR = vec4(col, 1.0);
}`;
    }

    /**
     * Get shader format metadata
     */
    getShaderFormat(target) {
        switch (target) {
            case 'glsl': return 'GLSL ES 3.0 (WebGL 2.0 compatible)';
            case 'hlsl_unreal': return 'HLSL Custom Expression for UE4/UE5';
            case 'unity_shadergraph': return 'Unity ShaderGraph JSON';
            case 'godot': return 'Godot GDShader 4.x';
            default: return 'Unknown';
        }
    }

    /**
     * Batch generate multiple variations
     */
    async batchGenerate(args) {
        const { base_params, variations, output_format = 'webp' } = args;

        // Get base parameters
        let baseParams = base_params;
        if (!baseParams && this.engine) {
            baseParams = this.engine.getAllParameters();
        }
        baseParams = baseParams || {};

        const results = [];
        let totalCredits = 0;

        for (const variation of variations) {
            const { vary, count, range } = variation;

            for (let i = 0; i < count; i++) {
                const variantParams = { ...baseParams };
                const progress = count > 1 ? i / (count - 1) : 0;

                // Apply variation
                switch (vary) {
                    case 'hue':
                        const hueRange = range || { min: 0, max: 360 };
                        variantParams.hue = hueRange.min + progress * (hueRange.max - hueRange.min);
                        break;
                    case 'geometry':
                        variantParams.geometry = Math.floor(progress * 23);
                        break;
                    case 'rotation':
                        variantParams.rot4dXW = progress * Math.PI * 2;
                        variantParams.rot4dYW = progress * Math.PI;
                        break;
                    case 'chaos':
                        const chaosRange = range || { min: 0, max: 1 };
                        variantParams.chaos = chaosRange.min + progress * (chaosRange.max - chaosRange.min);
                        break;
                    case 'intensity':
                        const intensityRange = range || { min: 0.3, max: 1.0 };
                        variantParams.intensity = intensityRange.min + progress * (intensityRange.max - intensityRange.min);
                        break;
                    case 'all':
                        variantParams.hue = Math.random() * 360;
                        variantParams.geometry = Math.floor(Math.random() * 24);
                        variantParams.chaos = Math.random();
                        variantParams.intensity = 0.3 + Math.random() * 0.7;
                        break;
                }

                // Apply to engine and render
                if (this.engine) {
                    for (const [key, value] of Object.entries(variantParams)) {
                        this.engine.setParameter(key, value);
                    }
                }

                results.push({
                    index: results.length,
                    variation_type: vary,
                    parameters: variantParams,
                    // In production: would generate actual image URL
                    preview_available: !!this.engine
                });

                // 0.8x credit multiplier for batch
                totalCredits += 0.8;
            }
        }

        telemetry.recordEvent(EventType.TOOL_INVOCATION_END, {
            tool: 'vib3_batch',
            count: results.length
        });

        return {
            count: results.length,
            output_format,
            variations: results,
            credits_used: Math.ceil(totalCredits),
            suggested_next_actions: ['render_preview', 'save_to_gallery']
        };
    }

    /**
     * Render current visualization to image
     */
    async renderPreview(args) {
        const { format = 'base64', width = 1024, height = 1024, include_metadata = true } = args;

        let imageData = null;
        let renderTime = 0;

        if (this.engine && this.engine.canvas) {
            const startTime = performance.now();

            try {
                // Render a frame
                this.engine.render();

                // Get canvas data
                const canvas = this.engine.canvas;
                if (format === 'base64' || format === 'png') {
                    imageData = canvas.toDataURL('image/png');
                } else if (format === 'webp') {
                    imageData = canvas.toDataURL('image/webp', 0.9);
                }

                renderTime = performance.now() - startTime;
            } catch (e) {
                console.warn('Render preview failed:', e);
            }
        }

        const params = this.engine ? this.engine.getAllParameters() : {};

        const result = {
            format,
            width,
            height,
            render_time_ms: renderTime,
            has_image: !!imageData
        };

        if (imageData && (format === 'base64' || format === 'png' || format === 'webp')) {
            result.data = imageData;
            result.mime_type = format === 'webp' ? 'image/webp' : 'image/png';
        }

        if (include_metadata) {
            result.parameters = params;
        }

        result.suggested_next_actions = ['vib3_export_shader', 'save_to_gallery', 'vib3_batch'];

        return result;
    }

    /**
     * Control animation playback
     */
    controlAnimation(args) {
        const { action, speed, choreography } = args;

        let animationState = {
            running: false,
            speed: 1.0,
            choreography: null
        };

        if (this.engine) {
            switch (action) {
                case 'start':
                    this.engine.start?.();
                    animationState.running = true;
                    break;
                case 'stop':
                    this.engine.stop?.();
                    animationState.running = false;
                    break;
                case 'toggle':
                    if (this.engine.isRunning) {
                        this.engine.stop?.();
                        animationState.running = false;
                    } else {
                        this.engine.start?.();
                        animationState.running = true;
                    }
                    break;
                case 'set_speed':
                    if (speed !== undefined) {
                        this.engine.setParameter('speed', speed);
                        animationState.speed = speed;
                    }
                    break;
            }

            if (choreography) {
                animationState.choreography = choreography;
                // Choreography would be applied via InteractivityManager
            }
        }

        return {
            action,
            animation_state: animationState,
            suggested_next_actions: ['render_preview', 'set_rotation']
        };
    }

    /**
     * Get API capabilities and pricing info
     */
    getCapabilities(args) {
        const { include_examples = false } = args;

        const capabilities = {
            version: '1.0.0',
            name: 'VIB3+ Procedural 4D Visualization API',
            description: 'Generate unique 4D geometric visualizations programmatically',

            systems: [
                { name: 'quantum', description: 'Organic, flowing 4D geometry with smooth transitions' },
                { name: 'faceted', description: 'Sharp, geometric patterns with clean edges' },
                { name: 'holographic', description: 'Multi-layer ethereal effects with depth' }
            ],

            geometries: {
                count: 24,
                base_types: BASE_GEOMETRIES.map(g => g.name),
                core_types: CORE_TYPES,
                encoding: 'geometry_index = core_index * 8 + base_index'
            },

            rotation: {
                dimensions: 6,
                planes: ['XY', 'XZ', 'YZ', 'XW', 'YW', 'ZW'],
                description: 'Full 6D rotation control. XW/YW/ZW are 4D hyperspace rotations.'
            },

            tools: Object.keys(toolDefinitions).map(name => ({
                name,
                description: toolDefinitions[name].description
            })),

            pricing: {
                model: 'Credit-based',
                tiers: [
                    { name: 'Free', credits: 50, price: '$0/month' },
                    { name: 'Starter', credits: 500, price: '$9/month' },
                    { name: 'Pro', credits: 5000, price: '$49/month' },
                    { name: 'Scale', credits: 50000, price: '$299/month' }
                ],
                credit_usage: {
                    'PNG 1024x1024': 1,
                    'PNG 4K': 4,
                    'WebP (any size)': 1,
                    'Shader export': 0,
                    'Batch (per image)': 0.8
                }
            },

            export_formats: ['png', 'webp', 'glsl', 'hlsl_unreal', 'unity_shadergraph', 'godot', 'json']
        };

        if (include_examples) {
            capabilities.examples = {
                quick_start: {
                    description: 'Generate a calm blue sphere',
                    tool: 'vib3_generate',
                    input: { prompt: 'calm blue sphere rotating gently' }
                },
                custom_geometry: {
                    description: 'Create specific geometry with parameters',
                    tool: 'create_4d_visualization',
                    input: { system: 'faceted', geometry_index: 9 }
                },
                shader_export: {
                    description: 'Export for Unreal Engine',
                    tool: 'vib3_export_shader',
                    input: { target: 'hlsl_unreal' }
                },
                batch_colors: {
                    description: 'Generate color variations',
                    tool: 'vib3_batch',
                    input: { variations: [{ vary: 'hue', count: 5 }] }
                }
            };
        }

        return capabilities;
    }
}

// Singleton instance
export const mcpServer = new MCPServer();
export default mcpServer;
