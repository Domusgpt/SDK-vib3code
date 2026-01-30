/**
 * VIB3+ Engine - Unified Visualization System
 * Coordinates Quantum, Faceted, and Holographic systems
 * Supports 24 geometries per system with full 6D rotation
 *
 * Backend selection: Faceted supports WebGL/WebGPU via UnifiedRenderBridge.
 * Quantum and Holographic use direct WebGL (5-layer canvas architecture).
 */

import { ParameterManager } from './Parameters.js';
import { CanvasManager } from './CanvasManager.js';
import { QuantumEngine } from '../quantum/QuantumEngine.js';
import { FacetedSystem } from '../faceted/FacetedSystem.js';
import { RealHolographicSystem } from '../holograms/RealHolographicSystem.js';
import { ReactivityManager } from '../reactivity/ReactivityManager.js';
import { ReactivityConfig } from '../reactivity/ReactivityConfig.js';

export class VIB3Engine {
    /**
     * @param {object} [options]
     * @param {boolean} [options.preferWebGPU=false] - Try WebGPU for supported systems
     * @param {boolean} [options.debug=false] - Enable debug logging
     * @param {object|ReactivityConfig} [options.reactivityConfig] - Initial reactivity config
     */
    constructor(options = {}) {
        this.activeSystem = null; // Only one system active at a time
        this.currentSystemName = 'quantum';
        this.parameters = new ParameterManager();
        this.initialized = false;
        this.canvasManager = null;

        /** @type {boolean} Whether to prefer WebGPU for systems that support it */
        this.preferWebGPU = options.preferWebGPU || false;

        /** @type {boolean} Debug mode */
        this.debug = options.debug || false;

        /** @type {ReactivityManager} Reactivity system for audio/tilt/interaction */
        this.reactivity = new ReactivityManager((name, value) => {
            this.parameters.setParameter(name, value);
            this.updateCurrentSystemParameters();
        });

        // Load initial reactivity config if provided
        if (options.reactivityConfig) {
            this.reactivity.loadConfig(options.reactivityConfig);
        }
    }

    /**
     * Initialize the VIB3+ engine
     */
    async initialize(containerId = 'vib3-container') {
        console.log('Initializing VIB3+ Engine');

        // Create CanvasManager
        try {
            this.canvasManager = new CanvasManager(containerId);
        } catch (error) {
            console.error('CanvasManager initialization failed:', error);
            return false;
        }

        // Initialize starting system
        await this.switchSystem(this.currentSystemName);

        // Sync base parameters to reactivity manager
        this.reactivity.setBaseParameters(this.parameters.getAllParameters());

        this.initialized = true;
        console.log('VIB3+ Engine initialized');
        return true;
    }

    /**
     * Create and initialize a specific system
     * CRITICAL: Engines find canvases by ID in DOM, not passed as parameters!
     */
    async createSystem(systemName) {
        console.log(`Creating ${systemName} system...`);

        // Create canvases with correct IDs in DOM
        const canvasIds = this.canvasManager.createSystemCanvases(systemName);

        let system = null;
        try {
            switch (systemName) {
                case 'quantum':
                    system = new QuantumEngine();
                    break;

                case 'faceted':
                    system = new FacetedSystem();

                    // Use bridge mode if WebGPU is preferred
                    if (this.preferWebGPU) {
                        const canvas = document.getElementById('content-canvas');
                        if (canvas) {
                            const bridgeSuccess = await system.initWithBridge(canvas, {
                                preferWebGPU: true,
                                debug: this.debug
                            });
                            if (bridgeSuccess) {
                                break;
                            }
                            // Fall through to direct WebGL if bridge fails
                            console.warn('Faceted bridge init failed, using direct WebGL');
                            system = new FacetedSystem();
                        }
                    }

                    const facetedSuccess = system.initialize();
                    if (!facetedSuccess) {
                        throw new Error('Faceted system initialization failed');
                    }
                    break;

                case 'holographic':
                    system = new RealHolographicSystem();
                    break;

                default:
                    throw new Error(`Unknown system: ${systemName}`);
            }

            // Register WebGL contexts with CanvasManager for cleanup
            canvasIds.forEach(canvasId => {
                const canvas = document.getElementById(canvasId);
                if (canvas) {
                    const gl = canvas.getContext('webgl') || canvas.getContext('webgl2');
                    if (gl) {
                        this.canvasManager.registerContext(canvasId, gl);
                    }
                }
            });

            // Apply current parameters
            system.updateParameters(this.parameters.getAllParameters());
            system.setActive(true);

            console.log(`${systemName} system created and activated`);
            return system;

        } catch (error) {
            console.error(`${systemName} system creation failed:`, error);
            throw error;
        }
    }

    /**
     * Switch between visualization systems
     * DESTROYS old system and CREATES new system (proper WebGL cleanup)
     */
    async switchSystem(systemName) {
        if (!['quantum', 'faceted', 'holographic'].includes(systemName)) {
            console.error('Unknown system:', systemName);
            return false;
        }

        console.log(`Switching from ${this.currentSystemName} to ${systemName}`);

        // Destroy active system if exists
        if (this.activeSystem) {
            if (this.activeSystem.setActive) {
                this.activeSystem.setActive(false);
            }
            if (this.activeSystem.destroy) {
                this.activeSystem.destroy();
            }
            this.activeSystem = null;
        }

        // Create and initialize new system
        try {
            this.activeSystem = await this.createSystem(systemName);
            this.currentSystemName = systemName;
            console.log(`Switched to ${systemName} system`);
            return true;
        } catch (error) {
            console.error(`Failed to switch to ${systemName}:`, error);
            return false;
        }
    }

    /**
     * Get the active backend type for the current system
     * @returns {'webgl'|'webgpu'|'direct-webgl'|null}
     */
    getActiveBackendType() {
        if (this.activeSystem && this.activeSystem.getBackendType) {
            return this.activeSystem.getBackendType();
        }
        return null;
    }

    /**
     * Update parameters for active system
     */
    updateCurrentSystemParameters() {
        const params = this.parameters.getAllParameters();

        if (this.activeSystem && this.activeSystem.updateParameters) {
            this.activeSystem.updateParameters(params);
        }
    }

    /**
     * Update a single parameter
     */
    setParameter(name, value) {
        this.parameters.setParameter(name, value);
        this.reactivity.setBaseParameter(name, value);
        this.updateCurrentSystemParameters();
    }

    /**
     * Update multiple parameters
     */
    setParameters(params) {
        this.parameters.setParameters(params);
        this.reactivity.setBaseParameters(params);
        this.updateCurrentSystemParameters();
    }

    /**
     * Get current parameter value
     */
    getParameter(name) {
        return this.parameters.getParameter(name);
    }

    /**
     * Get all parameters
     */
    getAllParameters() {
        return this.parameters.getAllParameters();
    }

    /**
     * Randomize all parameters
     */
    randomizeAll() {
        this.parameters.randomizeAll();
        this.updateCurrentSystemParameters();
    }

    /**
     * Reset to defaults
     */
    resetAll() {
        this.parameters.resetToDefaults();
        this.updateCurrentSystemParameters();
    }

    /**
     * Get current system name
     */
    getCurrentSystem() {
        return this.currentSystemName;
    }

    /**
     * Get active system instance
     */
    getActiveSystemInstance() {
        return this.activeSystem;
    }

    // ========================================================================
    // Reactivity
    // ========================================================================

    /**
     * Get the ReactivityManager instance
     * @returns {ReactivityManager}
     */
    getReactivityManager() {
        return this.reactivity;
    }

    /**
     * Load a reactivity configuration
     * @param {object|ReactivityConfig} config
     * @returns {{ valid: boolean, errors: string[] }}
     */
    loadReactivityConfig(config) {
        return this.reactivity.loadConfig(config);
    }

    /**
     * Get current reactivity configuration
     * @returns {object}
     */
    getReactivityConfig() {
        return this.reactivity.getConfig();
    }

    /**
     * Start reactivity processing (audio/tilt/interaction)
     */
    startReactivity() {
        this.reactivity.setBaseParameters(this.parameters.getAllParameters());
        this.reactivity.start();
    }

    /**
     * Stop reactivity processing
     */
    stopReactivity() {
        this.reactivity.stop();
    }

    /**
     * Check if reactivity is active
     * @returns {boolean}
     */
    isReactivityActive() {
        return this.reactivity.isActive;
    }

    /**
     * Feed audio data into the reactivity system
     * @param {number} bass - Bass level 0-1
     * @param {number} mid - Mid level 0-1
     * @param {number} high - High level 0-1
     * @param {number} [energy] - Overall energy (computed if omitted)
     */
    setAudioInput(bass, mid, high, energy) {
        this.reactivity.setAudioInput(bass, mid, high, energy);
    }

    /**
     * Feed device tilt data into the reactivity system
     * @param {number} alpha - Z-axis rotation (0-360)
     * @param {number} beta - X-axis rotation (-180 to 180)
     * @param {number} gamma - Y-axis rotation (-90 to 90)
     */
    setTiltInput(alpha, beta, gamma) {
        this.reactivity.setTiltInput(alpha, beta, gamma);
    }

    /**
     * Feed mouse position into the reactivity system
     * @param {number} x - Normalized X (0-1)
     * @param {number} y - Normalized Y (0-1)
     * @param {number} [velocityX=0]
     * @param {number} [velocityY=0]
     */
    setMouseInput(x, y, velocityX = 0, velocityY = 0) {
        this.reactivity.setMouseInput(x, y, velocityX, velocityY);
    }

    /**
     * Trigger a click event in the reactivity system
     * @param {number} [intensity=1.0]
     */
    triggerClick(intensity = 1.0) {
        this.reactivity.triggerClick(intensity);
    }

    /**
     * Feed scroll delta into the reactivity system
     * @param {number} delta
     */
    setScrollDelta(delta) {
        this.reactivity.setScrollDelta(delta);
    }

    /**
     * Feed touch data into the reactivity system
     * @param {Touch[]} touches
     * @param {number} [pinchScale=1]
     * @param {number} [rotation=0]
     */
    setTouchInput(touches, pinchScale = 1, rotation = 0) {
        this.reactivity.setTouchInput(touches, pinchScale, rotation);
    }

    // ========================================================================
    // Geometry
    // ========================================================================

    /**
     * Get geometry names for current system
     */
    getGeometryNames() {
        return [
            // Base geometries (0-7)
            'Tetrahedron',
            'Hypercube',
            'Sphere',
            'Torus',
            'Klein Bottle',
            'Fractal',
            'Wave',
            'Crystal',
            // Hypersphere Core (8-15)
            'Hypersphere Core (Tetrahedron)',
            'Hypersphere Core (Hypercube)',
            'Hypersphere Core (Sphere)',
            'Hypersphere Core (Torus)',
            'Hypersphere Core (Klein)',
            'Hypersphere Core (Fractal)',
            'Hypersphere Core (Wave)',
            'Hypersphere Core (Crystal)',
            // Hypertetrahedron Core (16-23)
            'Hypertetrahedron Core (Tetrahedron)',
            'Hypertetrahedron Core (Hypercube)',
            'Hypertetrahedron Core (Sphere)',
            'Hypertetrahedron Core (Torus)',
            'Hypertetrahedron Core (Klein)',
            'Hypertetrahedron Core (Fractal)',
            'Hypertetrahedron Core (Wave)',
            'Hypertetrahedron Core (Crystal)'
        ];
    }

    /**
     * Export current state
     */
    exportState() {
        return {
            system: this.currentSystemName,
            parameters: this.parameters.getAllParameters(),
            reactivity: this.reactivity.getConfig(),
            reactivityActive: this.reactivity.isActive,
            backend: this.getActiveBackendType(),
            timestamp: new Date().toISOString(),
            version: '1.1.0'
        };
    }

    /**
     * Import state
     */
    async importState(state) {
        if (!state || typeof state !== 'object') {
            console.warn('VIB3Engine: importState received invalid state');
            return;
        }
        if (state.system && typeof state.system === 'string') {
            await this.switchSystem(state.system);
        }
        if (state.parameters && typeof state.parameters === 'object') {
            this.parameters.setParameters(state.parameters);
            this.reactivity.setBaseParameters(state.parameters);
            this.updateCurrentSystemParameters();
        }
        if (state.reactivity && typeof state.reactivity === 'object') {
            this.reactivity.loadConfig(state.reactivity);
        }
        if (state.reactivityActive) {
            this.startReactivity();
        }
    }

    /**
     * Destroy engine and clean up
     */
    destroy() {
        // Stop and destroy reactivity
        if (this.reactivity) {
            this.reactivity.destroy();
        }

        // Destroy active system
        if (this.activeSystem && this.activeSystem.destroy) {
            this.activeSystem.destroy();
        }
        this.activeSystem = null;

        // Destroy canvas manager
        if (this.canvasManager) {
            this.canvasManager.destroy();
        }
        this.canvasManager = null;

        this.initialized = false;
        console.log('VIB3+ Engine destroyed');
    }
}
