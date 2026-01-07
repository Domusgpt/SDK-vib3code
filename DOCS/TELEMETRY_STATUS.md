# Telemetry and CLI Gap Analysis

This note answers whether the current telemetry output formats and command-line usability "have all the things they can and should" for the VIB3+ engine. In short: the codebase exposes telemetry module entry points in `package.json`, but the referenced implementations do **not** exist and no CLI wiring is present. The sections below summarize the evidence and what is missing, then outline the recommended output formats and command affordances to make the telemetry surface actually usable.

## Findings

1. **Missing telemetry modules referenced in exports**  
   The package exports include `./product/telemetry/facade` and `./product/telemetry/reference-providers`, but the corresponding source files are absent in `src/` (and in `types/`). Attempting to import these paths will fail at runtime.

2. **No in-repo telemetry references or commands**  
   A repo-wide search shows no telemetry-related code in `src/`, `js/`, or styles, and there are no CLI commands or scripts to emit or configure telemetry.

3. **Design intent noted but unimplemented**  
   The roadmap in `CLAUDE.md` lists an `EventEmitter` for telemetry and mentions a telemetry/CLI integration, but no implementation landed.

## Evidence

- `package.json` exports telemetry entry points that point to missing files. These paths do not exist under `src/` or elsewhere in the repo.  
- A search for the term "telemetry" returns only dependency metadata (e.g., Storybook packages) and no project code.
- `CLAUDE.md` documents telemetry goals without a backing module in the tree.

## What should be added (recommendation)

To satisfy the stated goals and make telemetry usable, the following pieces are needed:

### 1) Minimal telemetry facade and providers
- **Facade**: A small module that normalizes event objects and forwards them to providers (e.g., console logger, HTTP endpoint, file sink). Include schema validation to keep outputs consistent.
- **Reference providers**: Ship at least two defaults: (a) a console logger for local development and (b) a JSONL file sink (for agent batch analysis). Stub an HTTP/postMessage provider if the viewer runs in an iframe.

**Recommended payload shape (JSON):**
```json
{
  "event": "slider-change" | "gallery-load" | "export" | "error",
  "timestamp": "2024-07-22T12:34:56.789Z",
  "sessionId": "uuid",
  "context": {
    "system": "quantum" | "faceted" | "holographic",
    "geometry": "hypercube" | "rhombic-dodeca" | "…",
    "variation": 0,
    "controls": {
      "rot4dXY": 0.12,
      "rot4dXZ": 0.34,
      "rot4dYZ": 0.56,
      "rot4dXW": 0.78,
      "rot4dYW": 0.9,
      "rot4dZW": 1.2,
      "audioReactive": true,
      "bgAudioGain": 0.5
    }
  },
  "metrics": {
    "frameTimeMs": 14.8,
    "exportDurationMs": 220
  },
  "meta": {
    "version": "1.0.0",
    "userAgent": "VIB3+ CLI/1.0"
  }
}
```

### 2) CLI affordances
Add a lightweight command surface so agents can emit and inspect telemetry without modifying UI code:
- `npm run telemetry:emit -- --event slider-change --system quantum --geometry hypercube --rot4dXY 0.5 …`
- `npm run telemetry:tail` to stream JSONL output from the local sink.
- `npm run telemetry:validate <path>` to validate captured logs against the payload schema.

### 3) Output formats
- **JSONL primary**: One event per line, with the payload shape above, for easy ingestion and grepping.
- **NDJSON over stdout**: When running CLI commands, default to stdout with a `--file <path>` option for writing to disk.
- **Structured console**: Colorized, truncated view for quick debugging (e.g., `event | ts | system | geometry | metric summary`).

### 4) Integration points
- Hook the facade into the existing control handlers (slider changes, gallery loads, exports) so telemetry fires on meaningful user actions.
- Ensure the Viewer portal (once implemented) emits telemetry for device-orientation events and export completion.

## Acceptance checklist
- Imports of `./product/telemetry/facade` and `./product/telemetry/reference-providers` resolve to real modules with the payload schema and provider defaults above.
- CLI scripts exist for emit/tail/validate, documented in `README.md`.
- Telemetry outputs default to JSONL with schema validation and fail-safe logging when a sink is unavailable.
- Control and export pathways emit at least: slider changes, gallery loads, variation saves/loads, export start/finish, and error events.
