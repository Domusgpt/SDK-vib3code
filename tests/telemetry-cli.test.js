import { describe, expect, it } from 'vitest';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';

const __filename = fileURLToPath(import.meta.url);
const repoRoot = join(dirname(__filename), '..');

const runCli = (args) =>
  spawnSync('node', ['tools/telemetry/cli.js', ...args], {
    cwd: repoRoot,
    encoding: 'utf8'
  });

describe('telemetry CLI', () => {
  it('analyzes packs headlessly for automation coverage', () => {
    const result = runCli(['analyze-pack', 'tests/fixtures/automation-pack-valid.json', 'fixtures-demo']);

    expect(result.status).toBe(0);
    expect(result.stdout).toContain('States: 2, sequences: 1, rules: 1');
    expect(result.stdout).toContain('Steps: 2, estimated duration: 450ms');
    expect(result.stdout).toContain('modulators: 1, triggers: 1');
    expect(result.stdout).toContain('✅ pack analyzed');
  });

  it('surfaces lint failures with a non-zero exit code', () => {
    const result = runCli(['lint-pack', 'tests/fixtures/automation-pack-invalid.json', 'fixtures-bad']);

    expect(result.status).toBe(1);
    expect(result.stderr).toContain("Sequence 'broken' step 0 references missing state 'missing'");
    expect(result.stderr).toContain("Rule no-action references unknown state 'missing'");
  });

  it('exports packs into asset manifests with transitions', () => {
    const tempDir = mkdtempSync(join(tmpdir(), 'telemetry-export-'));
    const outputPath = join(tempDir, 'export.json');

    const result = runCli(['export-pack', 'tests/fixtures/automation-pack-valid.json', 'fixtures-demo', outputPath]);

    expect(result.status).toBe(0);
    expect(result.stdout).toContain('pack exported');

    const exported = JSON.parse(readFileSync(outputPath, 'utf8'));
    expect(exported.pack).toBe('fixtures-demo');
    expect(exported.assets).toHaveLength(2);
    expect(exported.transitions.length).toBeGreaterThan(0);

    rmSync(tempDir, { recursive: true, force: true });
  });
});
