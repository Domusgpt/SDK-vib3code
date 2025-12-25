import { VirtualLayerCompositor, DEFAULT_LAYERS } from '../../src/core/VirtualLayerCompositor.js';
import { WireframeRenderer } from '../../src/core/WireframeRenderer.js';
import { BackgroundGradient } from '../../src/core/BackgroundGradient.js';
import { PulseRings } from '../../src/core/PulseRings.js';
import { HighlightStreaks } from '../../src/core/HighlightStreaks.js';
import { ShadowVignette } from '../../src/core/ShadowVignette.js';
import { AudioChoreographer } from '../../src/core/AudioChoreographer.js';

class ReactiveInputs {
    constructor() {
        this.choreographer = new AudioChoreographer({ lookaheadMs: 200, fftSize: 256, smoothing: 0.8 });
    }

    async start() {
        await this.choreographer.start({ preferMic: true });
    }

    async enableMic() {
        await this.choreographer.startWithMicrophone();
    }

    async enableOscillator() {
        await this.choreographer.startWithOscillator();
    }

    async loadFile(file) {
        await this.choreographer.startWithFile(file);
    }

    sample() {
        return this.choreographer.sampleFrame();
    }

    get state() {
        return this.choreographer.state;
    }

    setTelemetry(payload = {}) {
        this.choreographer.setTelemetry(payload);
    }

    setSequenceMode(mode) {
        this.choreographer.setSequenceMode(mode);
    }
}

function createLayerPatterns(state, wireframeRenderer, backgroundGradient, pulseRings, highlightStreaks, shadowVignette) {
    const lerp = (a, b, t) => a + (b - a) * t;

    return {
        background: (gl, target, time) => {
            backgroundGradient.render(target, time, state.bands, state.energy);
        },
        shadow: (gl, target, time) => {
            shadowVignette.render(target, time, state.bands, state.energy, state.telemetry);
        },
        content: (gl, target, time) => {
            if (!wireframeRenderer) return;
            wireframeRenderer.render(target, time, state.bands, state.energy, state.telemetry);
        },
        highlight: (gl, target, time) => {
            highlightStreaks.render(target, time, state.bands, state.energy, state.telemetry);
        },
        accent: (gl, target, time) => {
            pulseRings.render(target, time, state.bands, state.energy, state.telemetry);
        }
    };
}

function bootUnifiedDemo() {
    const inputs = new ReactiveInputs();
    const compositor = new VirtualLayerCompositor({ containerId: 'canvasContainer', enableDepth: true });
    compositor.defineLayers(DEFAULT_LAYERS);

    const wireframeRenderer = new WireframeRenderer(compositor.gl);
    const backgroundGradient = new BackgroundGradient(compositor.gl);
    const pulseRings = new PulseRings(compositor.gl);
    const highlightStreaks = new HighlightStreaks(compositor.gl);
    const shadowVignette = new ShadowVignette(compositor.gl);
    const patterns = createLayerPatterns(
        inputs.state,
        wireframeRenderer,
        backgroundGradient,
        pulseRings,
        highlightStreaks,
        shadowVignette
    );

    inputs.start().then(() => {
        compositor.start(patterns, {
            beforeFrame: () => inputs.sample(),
            afterFrame: () => {
                const decay = 0.18;
                const energyImpulse = inputs.state.sequence?.intensity ?? inputs.state.energy;
                compositor.layers.forEach((layer) => {
                    const base = layer.name === 'background' ? 1.0 : 0.75;
                    const impulse = energyImpulse * (layer.name === 'accent' ? 1.5 : 1.0);
                    layer.alpha = Math.max(base - decay, Math.min(1.4, base + impulse));
                });

                const zone = inputs.state.telemetry.zone;
                if (zone === 'combat') wireframeRenderer.setGeometry('tesseract');
                if (zone === 'calm') wireframeRenderer.setGeometry('cube');
                const health = inputs.state.telemetry.player_health ?? 1;
                const combo = inputs.state.telemetry.combo_multiplier ?? 0;
                const snapBoost = inputs.state.sequence?.snap ?? 0;
                wireframeRenderer.setLineWidth(0.9 + (1 - health) * 1.8 + combo * 0.15 + snapBoost * 0.5);
            }
        });
    });

    window.vibUnifiedDemo = {
        compositor,
        inputs,
        wireframeRenderer,
        stop: () => compositor.stop(),
        start: () => compositor.start(patterns, { beforeFrame: () => inputs.sample() }),
        diagnostics: () => compositor.getDiagnostics(),
        reinitialize: (layers = DEFAULT_LAYERS) => {
            compositor.defineLayers(layers);
            compositor.start(patterns, { beforeFrame: () => inputs.sample() });
            return compositor.getDiagnostics();
        },
        pushTelemetry: (payload) => inputs.setTelemetry(payload),
        setGeometry: (type) => wireframeRenderer.setGeometry(type),
        setLineWidth: (width) => wireframeRenderer.setLineWidth(width),
        setSequenceMode: (mode) => inputs.setSequenceMode(mode),
        loadFile: (file) => inputs.loadFile(file)
    };

    console.log('🧪 Phase 2 unified canvas demo ready', compositor.getDiagnostics());
}

if (document.readyState === 'complete' || document.readyState === 'interactive') {
    bootUnifiedDemo();
} else {
    document.addEventListener('DOMContentLoaded', bootUnifiedDemo);
}
