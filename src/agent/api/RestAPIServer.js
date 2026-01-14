/**
 * VIB3+ REST API Server
 * Express-compatible server for GPT Actions and general REST API access
 *
 * This module provides:
 * - REST endpoints matching OpenAPI spec for GPT Store
 * - API key authentication
 * - Rate limiting
 * - CORS support for cross-origin requests
 */

import { mcpServer } from '../mcp/MCPServer.js';
import { telemetry, EventType } from '../telemetry/index.js';

/**
 * Credit usage per operation
 */
const CREDIT_COSTS = {
    'png_1024': 1,
    'png_2048': 2,
    'png_4096': 4,
    'webp': 1,
    'svg': 2,
    'shader': 0,
    'batch_multiplier': 0.8
};

/**
 * Rate limit tiers (requests per hour)
 */
const RATE_LIMITS = {
    free: 10,
    starter: 100,
    pro: 500,
    scale: 2000,
    enterprise: Infinity
};

/**
 * REST API Server class
 * Can be used standalone or integrated with Express/Fastify
 */
export class RestAPIServer {
    constructor(options = {}) {
        this.mcpServer = options.mcpServer || mcpServer;
        this.apiKeys = new Map(); // In production: use Redis/database
        this.rateLimits = new Map();
        this.port = options.port || 3000;
    }

    /**
     * Create Express-compatible middleware handlers
     */
    getMiddleware() {
        return {
            cors: this.corsMiddleware.bind(this),
            auth: this.authMiddleware.bind(this),
            rateLimit: this.rateLimitMiddleware.bind(this),
            errorHandler: this.errorHandler.bind(this)
        };
    }

    /**
     * CORS middleware
     */
    corsMiddleware(req, res, next) {
        res.setHeader('Access-Control-Allow-Origin', '*');
        res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
        res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key');

        if (req.method === 'OPTIONS') {
            res.status(204).end();
            return;
        }
        next();
    }

    /**
     * Authentication middleware
     */
    authMiddleware(req, res, next) {
        const apiKey = req.headers['x-api-key'] || req.headers['authorization']?.replace('Bearer ', '');

        if (!apiKey) {
            // Allow free tier with rate limiting
            req.tier = 'free';
            req.credits = 50;
            next();
            return;
        }

        // Validate API key (in production: database lookup)
        const keyInfo = this.apiKeys.get(apiKey);
        if (!keyInfo) {
            res.status(401).json({
                error: {
                    type: 'AuthenticationError',
                    code: 'INVALID_API_KEY',
                    message: 'Invalid or expired API key',
                    suggestion: 'Get a valid API key at https://vib3.dev/api-keys'
                }
            });
            return;
        }

        req.tier = keyInfo.tier;
        req.credits = keyInfo.credits;
        req.userId = keyInfo.userId;
        next();
    }

    /**
     * Rate limiting middleware
     */
    rateLimitMiddleware(req, res, next) {
        const identifier = req.userId || req.ip || 'anonymous';
        const limit = RATE_LIMITS[req.tier] || RATE_LIMITS.free;

        const now = Date.now();
        const windowStart = now - 3600000; // 1 hour window

        let record = this.rateLimits.get(identifier);
        if (!record || record.windowStart < windowStart) {
            record = { windowStart: now, count: 0 };
        }

        record.count++;
        this.rateLimits.set(identifier, record);

        // Add rate limit headers
        res.setHeader('X-RateLimit-Limit', limit);
        res.setHeader('X-RateLimit-Remaining', Math.max(0, limit - record.count));
        res.setHeader('X-RateLimit-Reset', new Date(record.windowStart + 3600000).toISOString());

        if (record.count > limit) {
            res.status(429).json({
                error: {
                    type: 'RateLimitError',
                    code: 'RATE_LIMIT_EXCEEDED',
                    message: `Rate limit exceeded. ${limit} requests per hour for ${req.tier} tier.`,
                    retry_after: Math.ceil((record.windowStart + 3600000 - now) / 1000),
                    suggestion: 'Upgrade your plan or wait for the rate limit to reset'
                }
            });
            return;
        }

        next();
    }

    /**
     * Error handler middleware
     */
    errorHandler(err, req, res, next) {
        console.error('API Error:', err);

        telemetry.recordEvent(EventType.TOOL_INVOCATION_ERROR, {
            path: req.path,
            error: err.message
        });

        res.status(err.status || 500).json({
            error: {
                type: 'ServerError',
                code: err.code || 'INTERNAL_ERROR',
                message: err.message || 'An unexpected error occurred',
                request_id: req.id,
                retry_possible: err.status < 500
            }
        });
    }

    /**
     * Get route handlers for Express/Fastify
     */
    getRoutes() {
        return {
            // Health check
            'GET /health': this.healthCheck.bind(this),

            // Core generation endpoints
            'POST /v1/generate': this.handleGenerate.bind(this),
            'POST /v1/generate/natural': this.handleGenerateNatural.bind(this),

            // Parameter manipulation
            'POST /v1/scene': this.handleCreateScene.bind(this),
            'POST /v1/scene/rotation': this.handleSetRotation.bind(this),
            'POST /v1/scene/parameters': this.handleSetParameters.bind(this),
            'GET /v1/scene/state': this.handleGetState.bind(this),

            // System switching
            'POST /v1/system': this.handleSwitchSystem.bind(this),

            // Export endpoints
            'POST /v1/export/shader': this.handleExportShader.bind(this),
            'POST /v1/export/image': this.handleExportImage.bind(this),

            // Batch operations
            'POST /v1/batch': this.handleBatch.bind(this),

            // Discovery
            'GET /v1/geometries': this.handleListGeometries.bind(this),
            'GET /v1/capabilities': this.handleGetCapabilities.bind(this),
            'GET /v1/schema': this.handleGetSchema.bind(this)
        };
    }

    // ============================================================
    // ROUTE HANDLERS
    // ============================================================

    /**
     * Health check endpoint
     */
    async healthCheck(req, res) {
        res.json({
            status: 'healthy',
            version: '1.0.0',
            timestamp: new Date().toISOString()
        });
    }

    /**
     * POST /v1/generate - Main generation endpoint
     */
    async handleGenerate(req, res) {
        const startTime = performance.now();
        const { style, geometry, core, color, rotation, output } = req.body;

        try {
            // Map REST params to MCP tool params
            let geometryIndex = geometry || 0;
            if (typeof geometry === 'string') {
                const baseNames = ['tetrahedron', 'hypercube', 'sphere', 'torus', 'klein', 'fractal', 'wave', 'crystal'];
                const baseIndex = baseNames.indexOf(geometry.toLowerCase());
                if (baseIndex >= 0) {
                    const coreIndex = core === 'hypersphere' ? 1 : core === 'hypertetrahedron' ? 2 : 0;
                    geometryIndex = coreIndex * 8 + baseIndex;
                }
            }

            // Create visualization
            const result = await this.mcpServer.handleToolCall('create_4d_visualization', {
                system: style || 'quantum',
                geometry_index: geometryIndex
            });

            // Set rotation if provided
            if (rotation) {
                await this.mcpServer.handleToolCall('set_rotation', {
                    XW: rotation.xw || 0,
                    YW: rotation.yw || 0,
                    ZW: rotation.zw || 0
                });
            }

            // Set color if provided
            if (color) {
                await this.mcpServer.handleToolCall('set_visual_parameters', {
                    hue: color.hue,
                    saturation: color.saturation,
                    intensity: color.intensity
                });
            }

            // Render preview
            const preview = await this.mcpServer.handleToolCall('render_preview', {
                format: output?.format || 'base64',
                width: output?.width || 1024,
                height: output?.height || 1024
            });

            const duration = performance.now() - startTime;

            // Calculate credits
            const credits = this.calculateCredits(output);

            res.json({
                success: true,
                url: preview.data ? null : `https://cdn.vib3.dev/${result.scene_id}.png`, // Placeholder CDN URL
                data: preview.data,
                expires_at: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
                parameters: result.geometry,
                render_time_ms: duration,
                credits_used: credits
            });

        } catch (error) {
            res.status(400).json({
                error: {
                    type: 'GenerationError',
                    code: 'GENERATION_FAILED',
                    message: error.message,
                    suggestion: 'Check parameter values and try again'
                }
            });
        }
    }

    /**
     * POST /v1/generate/natural - Natural language generation
     */
    async handleGenerateNatural(req, res) {
        const { prompt, output_format } = req.body;

        if (!prompt) {
            res.status(400).json({
                error: {
                    type: 'ValidationError',
                    code: 'MISSING_PROMPT',
                    message: 'The "prompt" field is required',
                    suggestion: 'Provide a natural language description of the visualization'
                }
            });
            return;
        }

        try {
            const result = await this.mcpServer.handleToolCall('vib3_generate', {
                prompt,
                output_format: output_format || 'png'
            });

            res.json({
                success: true,
                url: result.data ? null : `https://cdn.vib3.dev/generated_${Date.now()}.png`,
                data: result.data,
                interpreted_as: result.interpreted_as,
                original_prompt: prompt,
                credits_used: 1
            });

        } catch (error) {
            res.status(400).json({
                error: {
                    type: 'GenerationError',
                    code: 'NATURAL_GENERATION_FAILED',
                    message: error.message
                }
            });
        }
    }

    /**
     * POST /v1/scene - Create a new scene
     */
    async handleCreateScene(req, res) {
        const result = await this.mcpServer.handleToolCall('create_4d_visualization', req.body);
        res.json(result);
    }

    /**
     * POST /v1/scene/rotation - Set rotation
     */
    async handleSetRotation(req, res) {
        const result = await this.mcpServer.handleToolCall('set_rotation', req.body);
        res.json(result);
    }

    /**
     * POST /v1/scene/parameters - Set visual parameters
     */
    async handleSetParameters(req, res) {
        const result = await this.mcpServer.handleToolCall('set_visual_parameters', req.body);
        res.json(result);
    }

    /**
     * GET /v1/scene/state - Get current state
     */
    async handleGetState(req, res) {
        const result = await this.mcpServer.handleToolCall('get_state', {});
        res.json(result);
    }

    /**
     * POST /v1/system - Switch visualization system
     */
    async handleSwitchSystem(req, res) {
        const result = await this.mcpServer.handleToolCall('switch_system', req.body);
        res.json(result);
    }

    /**
     * POST /v1/export/shader - Export as shader code
     */
    async handleExportShader(req, res) {
        const result = await this.mcpServer.handleToolCall('vib3_export_shader', req.body);

        // Set content type based on format
        if (req.query.raw === 'true') {
            res.setHeader('Content-Type', 'text/plain');
            res.send(result.code);
        } else {
            res.json(result);
        }
    }

    /**
     * POST /v1/export/image - Export rendered image
     */
    async handleExportImage(req, res) {
        const result = await this.mcpServer.handleToolCall('render_preview', req.body);

        if (req.query.raw === 'true' && result.data) {
            // Return raw image data
            const base64Data = result.data.replace(/^data:image\/\w+;base64,/, '');
            const buffer = Buffer.from(base64Data, 'base64');
            res.setHeader('Content-Type', result.mime_type || 'image/png');
            res.send(buffer);
        } else {
            res.json(result);
        }
    }

    /**
     * POST /v1/batch - Batch generation
     */
    async handleBatch(req, res) {
        const result = await this.mcpServer.handleToolCall('vib3_batch', req.body);
        res.json(result);
    }

    /**
     * GET /v1/geometries - List available geometries
     */
    async handleListGeometries(req, res) {
        const result = await this.mcpServer.handleToolCall('search_geometries', {
            core_type: req.query.core_type || 'all'
        });
        res.json(result);
    }

    /**
     * GET /v1/capabilities - Get API capabilities
     */
    async handleGetCapabilities(req, res) {
        const result = await this.mcpServer.handleToolCall('get_capabilities', {
            include_examples: req.query.examples === 'true'
        });
        res.json(result);
    }

    /**
     * GET /v1/schema - Get parameter schema
     */
    async handleGetSchema(req, res) {
        const result = await this.mcpServer.handleToolCall('get_parameter_schema', {});
        res.json(result);
    }

    // ============================================================
    // HELPER METHODS
    // ============================================================

    /**
     * Calculate credit cost for an operation
     */
    calculateCredits(output = {}) {
        const width = output.width || 1024;
        const format = output.format || 'png';

        if (format === 'webp') return CREDIT_COSTS.webp;
        if (format === 'svg') return CREDIT_COSTS.svg;

        if (width >= 4096) return CREDIT_COSTS.png_4096;
        if (width >= 2048) return CREDIT_COSTS.png_2048;
        return CREDIT_COSTS.png_1024;
    }

    /**
     * Register an API key
     */
    registerApiKey(apiKey, info) {
        this.apiKeys.set(apiKey, {
            tier: info.tier || 'starter',
            credits: info.credits || 500,
            userId: info.userId,
            createdAt: new Date()
        });
    }

    /**
     * Create Express app (if Express is available)
     */
    createExpressApp() {
        // Dynamic import to avoid requiring express as a dependency
        try {
            const express = require('express');
            const app = express();

            app.use(express.json());

            // Apply middleware
            const middleware = this.getMiddleware();
            app.use(middleware.cors);
            app.use(middleware.auth);
            app.use(middleware.rateLimit);

            // Register routes
            const routes = this.getRoutes();
            for (const [route, handler] of Object.entries(routes)) {
                const [method, path] = route.split(' ');
                app[method.toLowerCase()](path, handler);
            }

            app.use(middleware.errorHandler);

            return app;

        } catch (e) {
            console.warn('Express not available. Use getRoutes() for manual integration.');
            return null;
        }
    }

    /**
     * Start the server (standalone mode)
     */
    async start() {
        const app = this.createExpressApp();
        if (app) {
            return new Promise((resolve) => {
                const server = app.listen(this.port, () => {
                    console.log(`VIB3+ API Server running on port ${this.port}`);
                    resolve(server);
                });
            });
        }
        console.log('Express not available. Server not started.');
        return null;
    }
}

// Export singleton
export const restApiServer = new RestAPIServer();
export default restApiServer;
