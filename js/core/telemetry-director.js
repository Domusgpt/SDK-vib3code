import telemetry from './telemetry.js';
import { telemetryControls } from '../../src/product/telemetry/schema.js';

const ease = {
  linear: (t) => t,
  smooth: (t) => t * t * (3 - 2 * t)
};

const systems = ['faceted', 'quantum', 'holographic', 'polychora'];
const interactions = ['mouse', 'click', 'scroll'];
const audioSensitivities = ['low', 'medium', 'high'];
const audioVisualModes = ['color', 'geometry', 'movement'];

function setCheckbox(id, checked) {
  const el = document.getElementById(id);
  if (el) {
    el.checked = !!checked;
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }
}

function applyControls(controls = {}) {
  Object.entries(controls).forEach(([param, value]) => {
    const slider = document.getElementById(param);
    if (slider && typeof value !== 'undefined') {
      if (slider.type === 'checkbox') {
        slider.checked = !!value;
      } else {
        slider.value = value;
      }
    }
    if (slider?.type !== 'checkbox' && typeof window.updateParameter === 'function') {
      window.updateParameter(param, value);
    }
  });
}

function applyReactivity(reactivity = {}) {
  if (typeof window.toggleSystemReactivity !== 'function') return;
  systems.forEach((system) => {
    const config = reactivity[system];
    if (!config) return;
    interactions.forEach((interaction) => {
      if (typeof config[interaction] === 'boolean') {
        const checkboxId = `${system}${interaction.charAt(0).toUpperCase()}${interaction.slice(1)}`;
        setCheckbox(checkboxId, config[interaction]);
        window.toggleSystemReactivity(system, interaction, config[interaction]);
      }
    });
  });
}

function applyAudioModes(audio = {}) {
  if (typeof window.toggleAudioReactivity !== 'function') return;
  audioSensitivities.forEach((sensitivity) => {
    const modes = audio[sensitivity];
    if (!modes) return;
    audioVisualModes.forEach((mode) => {
      if (typeof modes[mode] === 'boolean') {
        const checkboxId = `${sensitivity}${mode.charAt(0).toUpperCase()}${mode.slice(1)}`;
        setCheckbox(checkboxId, modes[mode]);
        window.toggleAudioReactivity(sensitivity, mode, modes[mode]);
      }
    });
  });
}

function captureControls() {
  const controls = {};
  telemetryControls.forEach((param) => {
    const el = document.getElementById(param);
    if (!el) return;
    controls[param] = el.type === 'checkbox' ? el.checked : parseFloat(el.value);
  });
  return controls;
}

function captureReactivity() {
  const reactivity = {};
  systems.forEach((system) => {
    const config = {};
    interactions.forEach((interaction) => {
      const checkboxId = `${system}${interaction.charAt(0).toUpperCase()}${interaction.slice(1)}`;
      const el = document.getElementById(checkboxId);
      if (el) config[interaction] = el.checked;
    });
    if (Object.keys(config).length) reactivity[system] = config;
  });
  return reactivity;
}

function captureAudio() {
  const audio = {};
  audioSensitivities.forEach((sensitivity) => {
    const modes = {};
    audioVisualModes.forEach((mode) => {
      const checkboxId = `${sensitivity}${mode.charAt(0).toUpperCase()}${mode.slice(1)}`;
      const el = document.getElementById(checkboxId);
      if (el) modes[mode] = el.checked;
    });
    if (Object.keys(modes).length) audio[sensitivity] = modes;
  });
  return audio;
}

function activeGeometryIndex() {
  const active = document.querySelector('.geom-btn.active');
  return active ? Number(active.dataset.index) || 0 : 0;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

class TelemetryAutomationDirector {
  constructor() {
    this.states = new Map();
    this.activeSequence = null;
    this.commandUnsub = telemetry.onEvent((event) => this.handleTelemetryCommand(event));
    window.addEventListener('storage', (event) => {
      if (event.key === 'vib3-automation-command' && event.newValue) {
        try {
          const command = JSON.parse(event.newValue);
          this.handleTelemetryCommand({ event: 'automation-command', context: { automation: command } });
        } catch (error) {
          console.warn('⚠️ automation command parse failed', error.message);
        }
      }
    });
  }

  snapshot(name) {
    const state = {
      system: window.currentSystem || 'faceted',
      geometry: activeGeometryIndex(),
      controls: captureControls(),
      reactivity: captureReactivity(),
      audio: captureAudio()
    };
    this.states.set(name, state);
    telemetry.emit('automation-state-snapshot', {
      context: {
        system: state.system,
        geometry: state.geometry,
        automation: { state: name }
      }
    });
    return state;
  }

  defineState(name, state) {
    this.states.set(name, state);
    telemetry.emit('automation-state-define', {
      context: {
        system: state.system,
        geometry: state.geometry,
        automation: { state: name, rule: state.rule }
      }
    });
  }

  async applyState(name, options = {}) {
    const target = this.states.get(name);
    if (!target) throw new Error(`State ${name} not found`);

    const fromState = options.fromState ? this.states.get(options.fromState) : null;
    if (options.sweep && fromState) {
      return this.runSweep(fromState, target, { ...options.sweep, targetState: name });
    }

    if (target.system && target.system !== window.currentSystem && typeof window.switchSystem === 'function') {
      window.switchSystem(target.system);
      await delay(50);
    }

    if (typeof target.geometry === 'number' && typeof window.selectGeometry === 'function') {
      window.selectGeometry(target.geometry);
    }

    applyControls(target.controls);
    applyReactivity(target.reactivity);
    applyAudioModes(target.audio);

    telemetry.emit('automation-state-apply', {
      context: {
        system: target.system,
        geometry: target.geometry,
        controls: target.controls,
        reactivity: target.reactivity,
        automation: { state: name }
      }
    });
  }

  async runSweep(fromState, toState, options = {}) {
    const duration = options.durationMs || 3000;
    const easing = ease[options.easing] || ease.smooth;
    const startTime = performance.now();
    const startControls = fromState.controls || {};
    const endControls = toState.controls || {};

    telemetry.emit('automation-sweep-start', {
      context: {
        system: toState.system || window.currentSystem,
        geometry: toState.geometry,
        controls: endControls,
        automation: {
          state: options.fromState || options.state,
          targetState: options.targetState,
          easing: options.easing || 'smooth',
          durationMs: duration
        }
      }
    });

    const step = (now) => {
      const rawT = Math.min(1, (now - startTime) / duration);
      const t = easing(rawT);
      const interpolated = {};

      Object.keys({ ...startControls, ...endControls }).forEach((key) => {
        const start = typeof startControls[key] === 'number' ? startControls[key] : endControls[key];
        const end = typeof endControls[key] === 'number' ? endControls[key] : startControls[key];
        if (typeof start === 'number' && typeof end === 'number') {
          interpolated[key] = start + (end - start) * t;
        } else {
          interpolated[key] = rawT < 0.5 ? start : end;
        }
      });

      applyControls(interpolated);
      telemetry.emit('automation-sweep-step', {
        context: {
          system: toState.system || window.currentSystem,
          geometry: toState.geometry,
          controls: interpolated,
          automation: {
            state: options.fromState,
            targetState: options.targetState,
            easing: options.easing || 'smooth',
            durationMs: duration,
            progress: Number(rawT.toFixed(3))
          }
        }
      });

      if (rawT < 1) {
        requestAnimationFrame(step);
      } else {
        applyReactivity(toState.reactivity);
        applyAudioModes(toState.audio);
        telemetry.emit('automation-sweep-complete', {
          context: {
            system: toState.system || window.currentSystem,
            geometry: toState.geometry,
            controls: toState.controls,
            reactivity: toState.reactivity,
            automation: {
              state: options.fromState,
              targetState: options.targetState,
              easing: options.easing || 'smooth',
              durationMs: duration
            }
          }
        });
      }
    };

    requestAnimationFrame(step);
  }

  async runSequence(sequenceName, steps = []) {
    this.activeSequence = { name: sequenceName, cancelled: false };
    telemetry.emit('automation-sequence-start', {
      context: { automation: { sequence: sequenceName } }
    });

    for (let i = 0; i < steps.length; i++) {
      if (this.activeSequence.cancelled) break;
      const stepDef = steps[i];
      const fromState = i === 0 ? stepDef.fromState : steps[i - 1].state;
      await this.applyState(stepDef.state, {
        fromState,
        sweep: stepDef.sweep
      });
      if (stepDef.holdMs) {
        await delay(stepDef.holdMs);
      }
    }

    telemetry.emit('automation-sequence-complete', {
      context: { automation: { sequence: sequenceName, loop: false } }
    });
    this.activeSequence = null;
  }

  cancelSequence() {
    if (this.activeSequence) {
      this.activeSequence.cancelled = true;
      telemetry.emit('automation-sequence-cancel', {
        context: { automation: { sequence: this.activeSequence.name } }
      });
      this.activeSequence = null;
    }
  }

  handleTelemetryCommand(event) {
    if (event.event !== 'automation-command') return;
    const automation = event.context?.automation || {};
    const action = automation.action;

    if (action === 'snapshot' && automation.state) {
      this.snapshot(automation.state);
    } else if (action === 'apply-state' && automation.state) {
      this.applyState(automation.state, {
        fromState: automation.fromState,
        sweep: automation.durationMs
          ? { durationMs: automation.durationMs, easing: automation.easing }
          : undefined
      });
    } else if (action === 'sweep' && automation.state && automation.targetState) {
      const from = this.states.get(automation.state);
      const to = this.states.get(automation.targetState);
      if (from && to) {
        this.runSweep(from, to, {
          fromState: automation.state,
          targetState: automation.targetState,
          durationMs: automation.durationMs,
          easing: automation.easing
        });
      }
    } else if (action === 'sequence' && Array.isArray(automation.steps)) {
      this.runSequence(automation.sequence || 'ad-hoc', automation.steps);
    }
  }
}

const director = new TelemetryAutomationDirector();
window.telemetryDirector = director;
window.captureAutomationState = (name) => director.snapshot(name);
window.applyAutomationState = (name, options) => director.applyState(name, options);
window.runAutomationSequence = (name, steps) => director.runSequence(name, steps);

export default director;
