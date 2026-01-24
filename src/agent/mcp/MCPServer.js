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

    buildResponse(operation, data, options = {}) {
        const response = {
            success: options.success ?? true,
            operation,
            timestamp: new Date().toISOString(),
            duration_ms: options.duration_ms ?? 0,
            ...data
        };
        return this.validateToolResponse(response);
    }

    validateToolResponse(response) {
        const validation = schemaRegistry.validate('toolResponse', response);
        if (!validation.valid) {
            telemetry.recordEvent(EventType.TOOL_INVOCATION_ERROR, {
                tool: response.operation,
                error: 'TOOL_RESPONSE_INVALID'
            });
            return {
                ...response,
                validation_errors: validation.errors
            };
        }
        return response;
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
            return this.buildResponse(toolName, { error: validation.error }, {
                success: false,
                duration_ms: performance.now() - startTime
            });
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
                case 'get_sdk_context':
                    result = this.getSDKContext();
                    break;
                case 'verify_knowledge':
                    result = this.verifyKnowledge(args);
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

            return this.buildResponse(toolName, result, { duration_ms: duration });

        } catch (error) {
            const duration = performance.now() - startTime;
            telemetry.recordEvent(EventType.TOOL_INVOCATION_ERROR, {
                tool: toolName,
                duration_ms: duration,
                error: error.message
            });

            return this.buildResponse(toolName, {
                error: {
                    type: 'SystemError',
                    code: 'TOOL_EXECUTION_FAILED',
                    message: error.message,
                    suggestion: 'Check engine initialization and parameters',
                    retry_possible: true
                }
            }, { success: false, duration_ms: duration });
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
     * Get SDK context for agent onboarding
     */
    getSDKContext() {
        return {
            sdk_name: 'VIB3+ SDK',
            version: '1.9.0',
            purpose: 'General-purpose 4D rotation visualization SDK for plugins, extensions, wearables, and agentic use',

            quick_reference: {
                visualization_systems: 4,
                rotation_planes: 6,
                base_geometries: 8,
                core_warp_types: 3,
                total_geometries: 24,
                canvas_layers_per_system: 5
            },

            systems: [
                { name: 'quantum', description: 'Complex quantum lattice visualizations' },
                { name: 'faceted', description: 'Clean 2D geometric patterns' },
                { name: 'holographic', description: '5-layer audio-reactive holographic effects' },
                { name: 'polychora', description: '4D polytopes with glassmorphic effects and physics' }
            ],

            geometry_encoding: {
                formula: 'geometry_index = core_index * 8 + base_index',
                base_geometries: ['tetrahedron', 'hypercube', 'sphere', 'torus', 'klein_bottle', 'fractal', 'wave', 'crystal'],
                core_types: ['base (0)', 'hypersphere (1)', 'hypertetrahedron (2)'],
                example: 'geometry 10 = hypersphere(2) because 1*8+2=10 → hypersphere core + sphere base'
            },

            rotation_planes: {
                '3D_space': ['XY', 'XZ', 'YZ'],
                '4D_hyperspace': ['XW', 'YW', 'ZW'],
                range: '-6.28 to 6.28 radians (±2π)'
            },

            canvas_layers: ['background', 'shadow', 'content', 'highlight', 'accent'],

            knowledge_quiz: {
                description: 'Verify understanding by calling verify_knowledge with answers',
                questions: [
                    'Q1: How many rotation planes? (integer)',
                    'Q2: What is the geometry encoding formula? (string)',
                    'Q3: How many canvas layers per system? (integer)',
                    'Q4: List the 4 visualization systems (array)',
                    'Q5: List the 3 core warp types (array)',
                    'Q6: List the 8 base geometry types (array)'
                ]
            },

            suggested_next_actions: ['verify_knowledge', 'create_4d_visualization', 'search_geometries']
        };
    }

    /**
     * Verify agent knowledge of SDK
     */
    verifyKnowledge(answers) {
        const expected = {
            rotation_planes: 6,
            canvas_layers: 5,
            systems: ['quantum', 'faceted', 'holographic', 'polychora'],
            core_types: ['base', 'hypersphere', 'hypertetrahedron'],
            base_geometries: ['tetrahedron', 'hypercube', 'sphere', 'torus', 'klein_bottle', 'fractal', 'wave', 'crystal']
        };

        const results = {
            passed: true,
            score: 0,
            max_score: 6,
            details: []
        };

        // Check rotation planes
        if (answers.rotation_planes === expected.rotation_planes) {
            results.score++;
            results.details.push({ question: 'rotation_planes', correct: true });
        } else {
            results.passed = false;
            results.details.push({
                question: 'rotation_planes',
                correct: false,
                expected: expected.rotation_planes,
                received: answers.rotation_planes
            });
        }

        // Check canvas layers
        if (answers.canvas_layers === expected.canvas_layers) {
            results.score++;
            results.details.push({ question: 'canvas_layers', correct: true });
        } else {
            results.passed = false;
            results.details.push({
                question: 'canvas_layers',
                correct: false,
                expected: expected.canvas_layers,
                received: answers.canvas_layers
            });
        }

        // Check systems (array comparison)
        const systemsCorrect = answers.systems &&
            answers.systems.length === 4 &&
            expected.systems.every(s => answers.systems.includes(s));
        if (systemsCorrect) {
            results.score++;
            results.details.push({ question: 'systems', correct: true });
        } else {
            results.passed = false;
            results.details.push({
                question: 'systems',
                correct: false,
                expected: expected.systems,
                received: answers.systems
            });
        }

        // Check geometry formula (partial match)
        if (answers.geometry_formula) {
            const formulaLower = answers.geometry_formula.toLowerCase();
            if (formulaLower.includes('core') && formulaLower.includes('8') && formulaLower.includes('base')) {
                results.score++;
                results.details.push({ question: 'geometry_formula', correct: true });
            } else {
                results.passed = false;
                results.details.push({
                    question: 'geometry_formula',
                    correct: false,
                    expected: 'geometry = coreIndex * 8 + baseIndex',
                    received: answers.geometry_formula
                });
            }
        }

        // Check core types (optional but scored)
        if (answers.core_types && answers.core_types.length === 3) {
            results.score++;
            results.details.push({ question: 'core_types', correct: true });
        }

        // Check base geometries (optional but scored)
        if (answers.base_geometries && answers.base_geometries.length === 8) {
            results.score++;
            results.details.push({ question: 'base_geometries', correct: true });
        }

        results.percentage = Math.round((results.score / results.max_score) * 100);
        results.status = results.percentage >= 50 ? 'ready_to_proceed' : 'needs_review';

        if (results.status === 'needs_review') {
            results.suggestion = 'Review DOCS/SYSTEM_INVENTORY.md for complete SDK context';
        }

        results.suggested_next_actions = results.status === 'ready_to_proceed'
            ? ['create_4d_visualization', 'get_state']
            : ['get_sdk_context'];

        return results;
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
}

// Singleton instance
export const mcpServer = new MCPServer();
export default mcpServer;
