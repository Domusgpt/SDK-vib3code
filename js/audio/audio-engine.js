import { AudioChoreographer } from '../../src/core/AudioChoreographer.js';

// Global audio state flags - CRITICAL for system integration
window.audioEnabled = false;

const defaultReactiveState = () => ({
    bass: 0,
    mid: 0,
    high: 0,
    energy: 0,
    bands: new Float32Array(7),
    triBands: new Float32Array(3),
    projected: false,
    sequence: { heartbeat: 0, swell: 0, snap: 0, intensity: 0, mode: 'hybrid' },
    source: 'idle'
});

export class ChoreographyAudioEngine {
    constructor() {
        this.choreographer = new AudioChoreographer({ lookaheadMs: 200, fftSize: 256, smoothing: 0.8 });
        this.isActive = false;
        this.source = 'idle';
        this.frameHandle = null;
        window.audioReactive = defaultReactiveState();
        console.log('🎵 Audio Choreographer: Ready with lookahead + tri-band gating');
    }

    async init({ preferMic = true } = {}) {
        if (this.isActive) return true;
        try {
            if (preferMic) {
                await this.choreographer.startWithMicrophone();
                this.source = 'mic';
            } else {
                await this.choreographer.startWithOscillator();
                this.source = 'oscillator';
            }
        } catch (error) {
            console.warn('Mic unavailable; falling back to oscillator', error);
            await this.choreographer.startWithOscillator();
            this.source = 'oscillator';
        }

        this.markActive();
        return true;
    }

    async enableMic() {
        await this.choreographer.startWithMicrophone();
        this.source = 'mic';
        this.markActive();
    }

    async enableOscillator() {
        await this.choreographer.startWithOscillator();
        this.source = 'oscillator';
        this.markActive();
    }

    async loadFile(file) {
        if (!file) return;
        await this.choreographer.startWithFile(file);
        this.source = 'file';
        this.markActive();
    }

    setSequenceMode(mode) {
        this.choreographer.setSequenceMode(mode);
    }

    setTelemetry(payload) {
        this.choreographer.setTelemetry(payload);
    }

    markActive() {
        if (!this.isActive) {
            this.isActive = true;
            window.audioEnabled = true;
            this.startProcessing();
        }
    }

    startProcessing() {
        const process = () => {
            if (!this.isActive) return;
            const frame = this.choreographer.sampleFrame();
            if (frame) {
                const [bass = 0, mid = 0, high = 0] = frame.triBands || [];
                window.audioReactive = {
                    ...window.audioReactive,
                    bass,
                    mid,
                    high,
                    energy: frame.energy ?? 0,
                    bands: frame.bands ?? window.audioReactive.bands,
                    triBands: frame.triBands ?? window.audioReactive.triBands,
                    projected: frame.projected ?? false,
                    sequence: frame.sequence ?? window.audioReactive.sequence,
                    source: this.source
                };
            }
            this.frameHandle = requestAnimationFrame(process);
        };

        if (this.frameHandle) cancelAnimationFrame(this.frameHandle);
        process();
    }

    isAudioActive() {
        return this.isActive && window.audioEnabled;
    }

    getAudioLevels() {
        return window.audioReactive;
    }

    stop() {
        this.isActive = false;
        window.audioEnabled = false;
        if (this.frameHandle) cancelAnimationFrame(this.frameHandle);
        this.frameHandle = null;
        this.choreographer.stop();
        window.audioReactive = defaultReactiveState();
        console.log('🎵 Audio Choreographer: Stopped');
    }
}

export function setupAudioToggle() {
    window.toggleAudio = function() {
        const audioBtn = document.querySelector('[onclick="toggleAudio()"]');

        if (!window.audioEngine.isActive) {
            window.audioEngine.init().then(success => {
                if (success) {
                    if (audioBtn) {
                        audioBtn.style.background = 'linear-gradient(45deg, rgba(0, 255, 0, 0.3), rgba(0, 255, 0, 0.6))';
                        audioBtn.style.borderColor = '#00ff00';
                        audioBtn.title = 'Audio Reactivity: ON';
                    }
                    console.log('🎵 Audio Reactivity: ON (choreographer)');
                } else {
                    console.log('⚠️ Audio permission denied or not available');
                }
            });
        } else {
            const audioEnabled = !window.audioEnabled;
            window.audioEnabled = audioEnabled;

            if (audioBtn) {
                audioBtn.style.background = audioEnabled ?
                    'linear-gradient(45deg, rgba(0, 255, 0, 0.3), rgba(0, 255, 0, 0.6))' :
                    'linear-gradient(45deg, rgba(255, 0, 255, 0.1), rgba(255, 0, 255, 0.3))';
                audioBtn.style.borderColor = audioEnabled ? '#00ff00' : 'rgba(255, 0, 255, 0.3)';
                audioBtn.title = `Audio Reactivity: ${audioEnabled ? 'ON' : 'OFF'}`;
            }

            if (audioEnabled) {
                navigator.mediaDevices.getUserMedia({ audio: true }).catch(e => {
                    window.audioEnabled = false;
                    console.log('⚠️ Audio permission denied:', e.message);
                });
            }

            console.log(`🎵 Audio Reactivity: ${audioEnabled ? 'ON' : 'OFF'}`);
        }
    };
}

const audioEngine = new ChoreographyAudioEngine();
window.audioEngine = audioEngine;
window.audioSources = {
    enableMic: () => audioEngine.enableMic(),
    enableOscillator: () => audioEngine.enableOscillator(),
    loadFile: (file) => audioEngine.loadFile(file)
};

setupAudioToggle();

console.log('🎵 Audio Engine Module: Loaded with choreographer integration');