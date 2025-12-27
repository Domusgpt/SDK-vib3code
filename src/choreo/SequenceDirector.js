const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

class SequenceDirector {
  constructor({
    baseAlpha = { background: 0.95, shadow: 0.9, content: 1.0, highlight: 0.9, accent: 0.9 },
    anticipateBoost = 0.1,
    stroke = { min: 0.7, max: 3.2, base: 1.05, healthGain: 1.6, comboGain: 0.22, snapGain: 0.55 },
    gates = { heartbeat: 0.18, swell: 0.16, snap: 0.18 }
  } = {}) {
    this.baseAlpha = baseAlpha;
    this.anticipateBoost = anticipateBoost;
    this.stroke = stroke;
    this.gates = gates;
  }

  evaluate(audioState = {}, telemetry = {}) {
    const tri = audioState.triBands || new Float32Array([0, 0, 0]);
    const energy = audioState.energy ?? 0;
    const seq = audioState.sequence || {};
    const projected = Boolean(audioState.projected);

    const heartbeat = seq.heartbeat >= this.gates.heartbeat ? seq.heartbeat : 0;
    const swell = seq.swell >= this.gates.swell ? seq.swell : 0;
    const snap = seq.snap >= this.gates.snap ? seq.snap : 0;

    const anticipation = projected ? this.anticipateBoost : 0;
    const health = clamp(telemetry.player_health ?? 1, 0, 1);
    const combo = clamp(telemetry.combo_multiplier ?? 0, 0, 5);
    const zone = telemetry.zone || 'calm';

    const layerAlpha = {
      background: clamp(
        this.baseAlpha.background + tri[0] * 0.4 + energy * 0.12 + anticipation,
        0.4,
        1.6
      ),
      shadow: clamp(this.baseAlpha.shadow + (1 - health) * 0.4 + tri[0] * 0.1, 0.35, 1.4),
      content: clamp(this.baseAlpha.content + energy * 0.6 + heartbeat * 0.65 + anticipation, 0.5, 1.7),
      highlight: clamp(this.baseAlpha.highlight + snap * 1.0 + tri[2] * 0.25 + anticipation, 0.45, 1.8),
      accent: clamp(this.baseAlpha.accent + swell * 0.8 + snap * 0.6 + tri[1] * 0.25 + anticipation, 0.5, 1.8)
    };

    const strokeWidth = clamp(
      this.stroke.base + (1 - health) * this.stroke.healthGain + combo * this.stroke.comboGain + snap * this.stroke.snapGain,
      this.stroke.min,
      this.stroke.max
    );

    const geometry = zone === 'combat' || snap > 0.35 ? 'tesseract' : 'cube';

    return {
      layerAlpha,
      strokeWidth,
      geometry,
      projected,
      cues: { heartbeat, swell, snap, energy }
    };
  }
}

export { SequenceDirector };
