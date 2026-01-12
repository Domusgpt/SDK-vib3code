/**
 * VIB3+ Interactivity Manager
 * Handles touch, swipe, gyroscope, mouse, and choreography (parameter sweeps)
 *
 * @module InteractivityManager
 * @version 1.0.0
 */

export class InteractivityManager {
    constructor(canvasElement, onParameterUpdate) {
        this.canvas = canvasElement;
        this.onParameterUpdate = onParameterUpdate;

        // Touch/mouse state
        this.isDragging = false;
        this.lastTouch = { x: 0, y: 0 };
        this.touchStartTime = 0;
        this.velocity = { x: 0, y: 0 };

        // Gyroscope state
        this.gyroEnabled = false;
        this.gyroPermissionGranted = false;
        this.gyroBaseline = { alpha: 0, beta: 0, gamma: 0 };

        // Choreography state
        this.choreographyActive = false;
        this.choreographyTimeline = [];
        this.choreographyIndex = 0;
        this.choreographyStartTime = 0;

        // Behavior settings
        this.settings = {
            // Touch/mouse sensitivity
            dragSensitivity: 0.01,        // Rotation per pixel
            swipeMomentum: true,          // Continue motion after release
            momentumDecay: 0.95,          // Velocity decay per frame

            // Gyroscope sensitivity
            gyroSensitivity: 0.02,        // Rotation per degree of tilt
            gyroSmoothing: 0.1,           // Low-pass filter strength

            // Auto-animation
            autoAnimate: false,           // Enable idle animation
            autoAnimateSpeed: 0.5,        // Speed of auto animation
            autoAnimateDelay: 5000,       // ms before auto-animate starts

            // Choreography
            choreographyLoop: true,       // Loop choreography sequences
            choreographySmooth: true,     // Smooth interpolation between keyframes

            // Interaction mapping
            dragMapping: {
                horizontal: 'rot4dXW',    // What horizontal drag controls
                vertical: 'rot4dYW'       // What vertical drag controls
            },
            pinchMapping: 'morphFactor',  // What pinch/zoom controls
            gyroMapping: {
                beta: 'rot4dXY',          // Front/back tilt
                gamma: 'rot4dYZ'          // Left/right tilt
            }
        };

        // Idle tracking for auto-animate
        this.lastInteractionTime = Date.now();
        this.autoAnimateFrame = null;

        // Momentum animation
        this.momentumFrame = null;

        // Initialize
        this.bindEvents();
    }

    /**
     * Bind all input events
     */
    bindEvents() {
        if (!this.canvas) return;

        // Mouse events
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.canvas.addEventListener('mouseleave', this.onMouseUp.bind(this));
        this.canvas.addEventListener('wheel', this.onWheel.bind(this));

        // Touch events
        this.canvas.addEventListener('touchstart', this.onTouchStart.bind(this), { passive: false });
        this.canvas.addEventListener('touchmove', this.onTouchMove.bind(this), { passive: false });
        this.canvas.addEventListener('touchend', this.onTouchEnd.bind(this));

        // Double-tap/click for reset
        this.canvas.addEventListener('dblclick', this.onDoubleClick.bind(this));

        console.log('🎮 InteractivityManager: Events bound');
    }

    // =========================================================================
    // MOUSE HANDLERS
    // =========================================================================

    onMouseDown(e) {
        this.isDragging = true;
        this.lastTouch = { x: e.clientX, y: e.clientY };
        this.touchStartTime = Date.now();
        this.velocity = { x: 0, y: 0 };
        this.stopMomentum();
        this.recordInteraction();
    }

    onMouseMove(e) {
        if (!this.isDragging) return;

        const dx = e.clientX - this.lastTouch.x;
        const dy = e.clientY - this.lastTouch.y;

        // Update velocity for momentum
        this.velocity = { x: dx, y: dy };

        // Apply rotation based on drag mapping
        this.applyDragRotation(dx, dy);

        this.lastTouch = { x: e.clientX, y: e.clientY };
    }

    onMouseUp(e) {
        if (!this.isDragging) return;
        this.isDragging = false;

        // Check for swipe (fast motion)
        const elapsed = Date.now() - this.touchStartTime;
        const speed = Math.sqrt(this.velocity.x ** 2 + this.velocity.y ** 2);

        if (this.settings.swipeMomentum && speed > 5 && elapsed < 300) {
            this.startMomentum();
        }
    }

    onWheel(e) {
        e.preventDefault();
        this.recordInteraction();

        // Wheel controls morphFactor or zoom
        const delta = e.deltaY * -0.001;
        this.onParameterUpdate({
            [this.settings.pinchMapping]: delta
        }, 'delta');
    }

    onDoubleClick(e) {
        // Reset rotations to zero
        this.onParameterUpdate({
            rot4dXY: 0, rot4dXZ: 0, rot4dYZ: 0,
            rot4dXW: 0, rot4dYW: 0, rot4dZW: 0
        }, 'absolute');
        this.recordInteraction();
    }

    // =========================================================================
    // TOUCH HANDLERS
    // =========================================================================

    onTouchStart(e) {
        e.preventDefault();
        this.stopMomentum();
        this.recordInteraction();

        if (e.touches.length === 1) {
            // Single touch - rotation
            this.isDragging = true;
            this.lastTouch = {
                x: e.touches[0].clientX,
                y: e.touches[0].clientY
            };
            this.touchStartTime = Date.now();
        } else if (e.touches.length === 2) {
            // Two fingers - pinch zoom
            this.lastPinchDistance = this.getPinchDistance(e.touches);
        }
    }

    onTouchMove(e) {
        e.preventDefault();

        if (e.touches.length === 1 && this.isDragging) {
            const touch = e.touches[0];
            const dx = touch.clientX - this.lastTouch.x;
            const dy = touch.clientY - this.lastTouch.y;

            this.velocity = { x: dx, y: dy };
            this.applyDragRotation(dx, dy);

            this.lastTouch = { x: touch.clientX, y: touch.clientY };

        } else if (e.touches.length === 2) {
            // Pinch zoom
            const newDistance = this.getPinchDistance(e.touches);
            const delta = (newDistance - this.lastPinchDistance) * 0.01;

            this.onParameterUpdate({
                [this.settings.pinchMapping]: delta
            }, 'delta');

            this.lastPinchDistance = newDistance;
        }
    }

    onTouchEnd(e) {
        if (e.touches.length === 0) {
            this.isDragging = false;

            const elapsed = Date.now() - this.touchStartTime;
            const speed = Math.sqrt(this.velocity.x ** 2 + this.velocity.y ** 2);

            if (this.settings.swipeMomentum && speed > 5 && elapsed < 300) {
                this.startMomentum();
            }
        }
    }

    getPinchDistance(touches) {
        const dx = touches[0].clientX - touches[1].clientX;
        const dy = touches[0].clientY - touches[1].clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }

    // =========================================================================
    // DRAG ROTATION
    // =========================================================================

    applyDragRotation(dx, dy) {
        const { dragMapping, dragSensitivity } = this.settings;

        this.onParameterUpdate({
            [dragMapping.horizontal]: dx * dragSensitivity,
            [dragMapping.vertical]: dy * dragSensitivity
        }, 'delta');
    }

    // =========================================================================
    // MOMENTUM
    // =========================================================================

    startMomentum() {
        this.momentumFrame = requestAnimationFrame(this.updateMomentum.bind(this));
    }

    updateMomentum() {
        const { momentumDecay } = this.settings;

        // Apply velocity
        this.applyDragRotation(this.velocity.x, this.velocity.y);

        // Decay velocity
        this.velocity.x *= momentumDecay;
        this.velocity.y *= momentumDecay;

        // Continue if still moving
        const speed = Math.sqrt(this.velocity.x ** 2 + this.velocity.y ** 2);
        if (speed > 0.1) {
            this.momentumFrame = requestAnimationFrame(this.updateMomentum.bind(this));
        }
    }

    stopMomentum() {
        if (this.momentumFrame) {
            cancelAnimationFrame(this.momentumFrame);
            this.momentumFrame = null;
        }
    }

    // =========================================================================
    // GYROSCOPE
    // =========================================================================

    async enableGyroscope() {
        // Request permission on iOS 13+
        if (typeof DeviceOrientationEvent !== 'undefined' &&
            typeof DeviceOrientationEvent.requestPermission === 'function') {
            try {
                const permission = await DeviceOrientationEvent.requestPermission();
                if (permission !== 'granted') {
                    console.warn('Gyroscope permission denied');
                    return false;
                }
            } catch (e) {
                console.error('Gyroscope permission error:', e);
                return false;
            }
        }

        // Capture baseline orientation
        this.gyroBaseline = { alpha: 0, beta: 0, gamma: 0 };

        window.addEventListener('deviceorientation', this.onDeviceOrientation.bind(this));
        this.gyroEnabled = true;
        this.gyroPermissionGranted = true;

        console.log('🔄 Gyroscope enabled');
        return true;
    }

    disableGyroscope() {
        window.removeEventListener('deviceorientation', this.onDeviceOrientation.bind(this));
        this.gyroEnabled = false;
        console.log('🔄 Gyroscope disabled');
    }

    onDeviceOrientation(e) {
        if (!this.gyroEnabled) return;

        const { gyroSensitivity, gyroSmoothing, gyroMapping } = this.settings;

        // Get orientation relative to baseline
        const beta = (e.beta || 0) - this.gyroBaseline.beta;   // Front/back tilt (-180 to 180)
        const gamma = (e.gamma || 0) - this.gyroBaseline.gamma; // Left/right tilt (-90 to 90)

        // Apply with smoothing
        this.onParameterUpdate({
            [gyroMapping.beta]: beta * gyroSensitivity * gyroSmoothing,
            [gyroMapping.gamma]: gamma * gyroSensitivity * gyroSmoothing
        }, 'delta');

        this.recordInteraction();
    }

    calibrateGyroscope() {
        // Set current orientation as baseline
        window.addEventListener('deviceorientation', (e) => {
            this.gyroBaseline = {
                alpha: e.alpha || 0,
                beta: e.beta || 0,
                gamma: e.gamma || 0
            };
        }, { once: true });
    }

    // =========================================================================
    // CHOREOGRAPHY (Parameter Sweeps)
    // =========================================================================

    /**
     * Create a choreography sequence
     * @param {Array} keyframes - Array of { time: ms, params: {...}, easing: 'linear'|'easeIn'|'easeOut'|'easeInOut' }
     */
    createChoreography(keyframes) {
        this.choreographyTimeline = keyframes.sort((a, b) => a.time - b.time);
        console.log(`🎬 Choreography created with ${keyframes.length} keyframes`);
    }

    /**
     * Start choreography playback
     */
    startChoreography() {
        if (this.choreographyTimeline.length === 0) {
            console.warn('No choreography defined');
            return;
        }

        this.choreographyActive = true;
        this.choreographyStartTime = performance.now();
        this.choreographyIndex = 0;

        requestAnimationFrame(this.updateChoreography.bind(this));
        console.log('🎬 Choreography started');
    }

    /**
     * Stop choreography playback
     */
    stopChoreography() {
        this.choreographyActive = false;
        console.log('🎬 Choreography stopped');
    }

    updateChoreography() {
        if (!this.choreographyActive) return;

        const elapsed = performance.now() - this.choreographyStartTime;
        const timeline = this.choreographyTimeline;

        // Find current and next keyframe
        let currentKF = timeline[0];
        let nextKF = timeline[1];

        for (let i = 0; i < timeline.length - 1; i++) {
            if (elapsed >= timeline[i].time && elapsed < timeline[i + 1].time) {
                currentKF = timeline[i];
                nextKF = timeline[i + 1];
                break;
            }
        }

        // Check if we're past the end
        const lastKF = timeline[timeline.length - 1];
        if (elapsed >= lastKF.time) {
            if (this.settings.choreographyLoop) {
                // Loop back to start
                this.choreographyStartTime = performance.now();
            } else {
                this.stopChoreography();
                return;
            }
        }

        // Interpolate between keyframes
        if (this.settings.choreographySmooth && nextKF) {
            const duration = nextKF.time - currentKF.time;
            const progress = (elapsed - currentKF.time) / duration;
            const easedProgress = this.applyEasing(progress, nextKF.easing || 'easeInOut');

            const interpolated = {};
            for (const param in nextKF.params) {
                const from = currentKF.params[param] ?? 0;
                const to = nextKF.params[param];
                interpolated[param] = from + (to - from) * easedProgress;
            }

            this.onParameterUpdate(interpolated, 'absolute');
        }

        requestAnimationFrame(this.updateChoreography.bind(this));
    }

    applyEasing(t, easing) {
        switch (easing) {
            case 'linear': return t;
            case 'easeIn': return t * t;
            case 'easeOut': return t * (2 - t);
            case 'easeInOut': return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
            default: return t;
        }
    }

    // =========================================================================
    // PRESET CHOREOGRAPHIES
    // =========================================================================

    /**
     * Create common choreography presets
     */
    static presets = {
        // Gentle continuous rotation
        gentleSpin: [
            { time: 0, params: { rot4dXW: 0, rot4dYW: 0 }, easing: 'linear' },
            { time: 10000, params: { rot4dXW: 6.28, rot4dYW: 3.14 }, easing: 'linear' }
        ],

        // Dramatic reveal
        reveal: [
            { time: 0, params: { rot4dXW: 0, rot4dYW: 0, intensity: 0 }, easing: 'easeOut' },
            { time: 2000, params: { rot4dXW: 1.57, rot4dYW: 0.5, intensity: 1 }, easing: 'easeInOut' },
            { time: 5000, params: { rot4dXW: 3.14, rot4dYW: 1.0, intensity: 0.7 }, easing: 'easeIn' }
        ],

        // Pulsing intensity
        pulse: [
            { time: 0, params: { intensity: 0.3, morphFactor: 0.8 }, easing: 'easeInOut' },
            { time: 1000, params: { intensity: 1.0, morphFactor: 1.2 }, easing: 'easeInOut' },
            { time: 2000, params: { intensity: 0.3, morphFactor: 0.8 }, easing: 'easeInOut' }
        ],

        // Color sweep
        colorSweep: [
            { time: 0, params: { hue: 0 }, easing: 'linear' },
            { time: 5000, params: { hue: 180 }, easing: 'linear' },
            { time: 10000, params: { hue: 360 }, easing: 'linear' }
        ],

        // Full 4D rotation
        full4DRotation: [
            { time: 0, params: { rot4dXY: 0, rot4dXZ: 0, rot4dYZ: 0, rot4dXW: 0, rot4dYW: 0, rot4dZW: 0 }, easing: 'linear' },
            { time: 20000, params: { rot4dXY: 6.28, rot4dXZ: 6.28, rot4dYZ: 6.28, rot4dXW: 6.28, rot4dYW: 6.28, rot4dZW: 6.28 }, easing: 'linear' }
        ],

        // Chaos burst
        chaosBurst: [
            { time: 0, params: { chaos: 0, speed: 0.5 }, easing: 'easeIn' },
            { time: 500, params: { chaos: 1, speed: 3 }, easing: 'easeOut' },
            { time: 2000, params: { chaos: 0.2, speed: 1 }, easing: 'easeInOut' }
        ]
    };

    loadPresetChoreography(name) {
        const preset = InteractivityManager.presets[name];
        if (preset) {
            this.createChoreography(preset);
            return true;
        }
        console.warn(`Unknown choreography preset: ${name}`);
        return false;
    }

    // =========================================================================
    // AUTO-ANIMATE (Idle Animation)
    // =========================================================================

    recordInteraction() {
        this.lastInteractionTime = Date.now();
        this.stopAutoAnimate();
    }

    startAutoAnimate() {
        if (this.settings.autoAnimate && !this.autoAnimateFrame) {
            this.autoAnimateFrame = requestAnimationFrame(this.updateAutoAnimate.bind(this));
        }
    }

    stopAutoAnimate() {
        if (this.autoAnimateFrame) {
            cancelAnimationFrame(this.autoAnimateFrame);
            this.autoAnimateFrame = null;
        }
    }

    updateAutoAnimate() {
        const idle = Date.now() - this.lastInteractionTime;

        if (idle > this.settings.autoAnimateDelay) {
            // Apply gentle auto-rotation
            const speed = this.settings.autoAnimateSpeed * 0.001;
            this.onParameterUpdate({
                rot4dXW: speed,
                rot4dYW: speed * 0.7
            }, 'delta');
        }

        if (this.settings.autoAnimate) {
            this.autoAnimateFrame = requestAnimationFrame(this.updateAutoAnimate.bind(this));
        }
    }

    // =========================================================================
    // SETTINGS
    // =========================================================================

    updateSettings(newSettings) {
        Object.assign(this.settings, newSettings);
        console.log('🎮 InteractivityManager settings updated:', newSettings);
    }

    getSettings() {
        return { ...this.settings };
    }

    // =========================================================================
    // CLEANUP
    // =========================================================================

    destroy() {
        this.stopMomentum();
        this.stopAutoAnimate();
        this.stopChoreography();
        this.disableGyroscope();

        // Remove event listeners
        if (this.canvas) {
            this.canvas.removeEventListener('mousedown', this.onMouseDown);
            this.canvas.removeEventListener('mousemove', this.onMouseMove);
            this.canvas.removeEventListener('mouseup', this.onMouseUp);
            this.canvas.removeEventListener('wheel', this.onWheel);
            this.canvas.removeEventListener('touchstart', this.onTouchStart);
            this.canvas.removeEventListener('touchmove', this.onTouchMove);
            this.canvas.removeEventListener('touchend', this.onTouchEnd);
        }

        console.log('🎮 InteractivityManager destroyed');
    }
}

// Export singleton for global access
export const interactivityManager = new InteractivityManager(null, () => {});
