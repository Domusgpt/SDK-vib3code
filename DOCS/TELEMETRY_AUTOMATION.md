# Telemetry Automation & State Sweeps

The telemetry director bridges UI controls with automation-friendly state management so agents can define reusable states, sweep between them with easing, and author rule-based sequences that exceed the stock engine panel. It exposes a programmable surface on top of the existing telemetry bus.

## Quick start
- Snapshot the live UI (all sliders, reactivity toggles, audio cells, geometry):
  ```js
  captureAutomationState('calm-grid');
  ```
- Apply a saved state immediately (auto-switches system/geometry):
  ```js
  applyAutomationState('calm-grid');
  ```
- Sweep from one state into another over 4s with smooth easing:
  ```js
  applyAutomationState('neon-burst', { fromState: 'calm-grid', sweep: { durationMs: 4000, easing: 'smooth' } });
  ```
- Run a multi-step sequence with holds:
  ```js
  runAutomationSequence('tour', [
    { state: 'calm-grid', holdMs: 800 },
    { state: 'pulse', sweep: { durationMs: 1800 } },
    { state: 'neon-burst', sweep: { durationMs: 2200 }, holdMs: 1000 }
  ]);
  ```

## State payload
Each state stores:
- `system` and `geometry` (auto-applied via `switchSystem`/`selectGeometry`).
- **Controls**: every slider from `telemetryControls` (rotations, grid/morph/chaos/speed, color sliders, audio gain) plus reactivity/audio checkboxes.
- **Reactivity grid**: per-system mouse/click/scroll toggles mapped through `toggleSystemReactivity`.
- **Audio reactivity**: low/medium/high × color/geometry/movement checkboxes, wired to `toggleAudioReactivity`.

## Command ingress (agent-friendly)
Automation can be driven via telemetry events or `localStorage`:
- Emit a telemetry event with `event: "automation-command"` and a context payload:
  ```js
  window.telemetry.emit('automation-command', {
    context: {
      automation: {
        action: 'sweep',
        state: 'calm-grid',
        targetState: 'neon-burst',
        durationMs: 2500,
        easing: 'smooth'
      }
    }
  });
  ```
- Drop a JSON command into `localStorage.setItem('vib3-automation-command', '{"action":"apply-state","state":"pulse"}')` to trigger from outside the frame (storage listener picks it up).

Supported actions: `snapshot`, `apply-state`, `sweep`, and `sequence` (with `steps` array shaped like `runAutomationSequence`).

## Telemetry emitted
Automation actions emit structured events for downstream collection:
- `automation-state-snapshot`, `automation-state-define`, `automation-state-apply`
- `automation-sweep-start` / `automation-sweep-step` / `automation-sweep-complete`
- `automation-sequence-start` / `automation-sequence-complete` / `automation-sequence-cancel`

Each includes `context.automation` describing `state`, `targetState`, `sequence`, `durationMs`, `easing`, and sweep `progress` for step events, enabling replay or audit.

## Style-pack readiness
Because states capture every control and toggle, you can serialize `director.states` to ship multi-state presets inside style packs. Pair a style pack with a short automation sequence to animate transitions as part of asset-pack exports or onboarding demos.
