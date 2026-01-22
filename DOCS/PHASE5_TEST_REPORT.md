# Phase 5 test report

## Run metadata
- Run ID: e2e-2026-01-22-01
- Timestamp: 2026-01-22T19:08:44Z
- Commit hash: HEAD (see git rev-parse HEAD)
- Tester: GPT-5.2-Codex (automation)

## Coverage summary
- Unit + integration: ✅ `pnpm test`
- Agentic workflow: ⚠️ MCP tool sequence not executed (no running MCP server in environment)
- Export pipeline: ✅ `pnpm cli:telemetry -- tests/artifacts/phase5-pack.json --json --non-interactive --preview-count=4`
- Cross‑platform sanity: ⚠️ Not exercised in container
- XR benchmark: ⚠️ Not executed (GPU benchmark requires dedicated hardware)

## E2E runbook execution notes
1. **Web playground verification**
   - Started Vite dev server with `pnpm exec vite --host 0.0.0.0 --port 4173`.
   - Loaded UI in headless browser and captured a screenshot.
2. **MCP workflow**
   - Skipped: requires MCP server + agent orchestration not available in container.
3. **Telemetry export**
   - Generated export manifest + preview hashes via CLI (using local pack JSON).
4. **Gallery operations**
   - Skipped: requires interactive UI state and persistence beyond headless smoke.
5. **Export formats**
   - Covered via unit tests (`src/testing/exportFormats.test.js`).
6. **XR benchmark**
   - Skipped: requires XR runtime + GPU instrumentation.

## Key results
- Pass/fail summary: ✅ Core unit tests, lint, and telemetry export succeeded.
- Regressions detected: None observed.
- Notes:
  - AJV emitted `date-time` format warnings during metrics validation; schema validation still passed.
  - Vite attempted to auto-open a browser and reported `xdg-open` missing in container.

## Attachments
- Metrics JSON: `DOCS/PHASE5_METRICS_REPORT.json`
- Telemetry pack input: `tests/artifacts/phase5-pack.json`
- XR report: `DOCS/PHASE5_XR_BENCHMARK_REPORT.md`
- Screenshot: `artifacts/vib3-home.png`
