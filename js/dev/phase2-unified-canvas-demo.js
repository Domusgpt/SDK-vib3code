import { VirtualLayerCompositor, DEFAULT_LAYERS } from '../../src/core/VirtualLayerCompositor.js';
import { WireframeRenderer } from '../../src/core/WireframeRenderer.js';
import { BackgroundGradient } from '../../src/core/BackgroundGradient.js';
import { PulseRings } from '../../src/core/PulseRings.js';
import { HighlightStreaks } from '../../src/core/HighlightStreaks.js';
import { ShadowVignette } from '../../src/core/ShadowVignette.js';
import { AudioChoreographer } from '../../src/core/AudioChoreographer.js';
import { SequenceDirector } from '../../src/choreo/SequenceDirector.js';

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

function createLayerPatterns(getState, getCues, wireframeRenderer, backgroundGradient, pulseRings, highlightStreaks, shadowVignette) {
    const lerp = (a, b, t) => a + (b - a) * t;

    return {
        background: (gl, target, time) => {
            const state = getState();
            const cues = getCues();
            const energy = cues?.cues?.energy ?? state.energy;
            backgroundGradient.render(target, time, state.bands, energy);
        },
        shadow: (gl, target, time) => {
            const state = getState();
            shadowVignette.render(target, time, state.bands, state.energy, state.telemetry);
        },
        content: (gl, target, time) => {
            if (!wireframeRenderer) return;
            const state = getState();
            wireframeRenderer.render(target, time, state.bands, state.energy, state.telemetry);
        },
        highlight: (gl, target, time) => {
            const state = getState();
            const cues = getCues();
            const energy = cues?.cues?.snap ?? state.energy;
            highlightStreaks.render(target, time, state.bands, energy, state.telemetry);
        },
        accent: (gl, target, time) => {
            const state = getState();
            const cues = getCues();
            const energy = cues?.cues?.swell ?? state.energy;
            pulseRings.render(target, time, state.bands, energy, state.telemetry);
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
    const director = new SequenceDirector();
    let cues = director.evaluate(inputs.state, inputs.state.telemetry);

    const patterns = createLayerPatterns(
        () => inputs.state,
        () => cues,
        wireframeRenderer,
        backgroundGradient,
        pulseRings,
        highlightStreaks,
        shadowVignette
    );

    inputs.start().then(() => {
        compositor.start(patterns, {
            beforeFrame: () => {
                const sampled = inputs.sample();
                const state = sampled || inputs.state;
                cues = director.evaluate(state, state.telemetry);
                return sampled;
            },
            afterFrame: () => {
                const decay = 0.18;
                const layerAlpha = cues?.layerAlpha || {};
                compositor.layers.forEach((layer) => {
                    const fallback = layer.name === 'background' ? 1.0 : 0.75;
                    const target = layerAlpha[layer.name] ?? fallback;
                    layer.alpha = Math.max(fallback - decay, Math.min(1.6, target));
                });

                wireframeRenderer.setGeometry(cues.geometry);
                wireframeRenderer.setLineWidth(cues.strokeWidth);
            }
        });
    });

    window.vibUnifiedDemo = {
        compositor,
        inputs,
        director,
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
        loadFile: (file) => inputs.loadFile(file),
        getCues: () => cues
    };

    console.log('🧪 Phase 2 unified canvas demo ready', compositor.getDiagnostics());
}

if (document.readyState === 'complete' || document.readyState === 'interactive') {
    bootUnifiedDemo();
} else {
    document.addEventListener('DOMContentLoaded', bootUnifiedDemo);
}
