import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { test } from 'node:test';

const exec = promisify(execFile);
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');

test('packed HTTP SDK installs in an external ESM and CommonJS project', async t => {
  const work = await mkdtemp(join(tmpdir(), 'veya-client-package-'));
  t.after(() => rm(work, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 }));
  const npm = process.env.npm_execpath ?? join(dirname(process.execPath), 'node_modules', 'npm', 'bin', 'npm-cli.js');
  const npmRun = (args, cwd) => exec(process.execPath, [npm, ...args], { cwd,
    env: { ...process.env, npm_config_audit: 'false', npm_config_fund: 'false' } });
  const { stdout } = await npmRun(['pack', '--json', '--silent', '--pack-destination', work], root);
  const packReport = JSON.parse(stdout);
  const [packed] = Array.isArray(packReport) ? packReport : Object.values(packReport);
  assert.equal(packed?.name, 'veya-client');
  const files = new Set(packed.files.map(file => file.path));
  for (const file of ['LICENSE', 'README.md', 'dist/esm/index.js', 'dist/cjs/index.js', 'dist/esm/index.d.ts']) {
    assert(files.has(file), `packed SDK is missing ${file}`);
  }
  const consumer = join(work, 'consumer');
  await mkdir(consumer);
  await writeFile(join(consumer, 'package.json'), JSON.stringify({ private: true, type: 'module' }));
  await npmRun(['install', '--ignore-scripts', '--no-package-lock', join(work, packed.filename)], consumer);
  await writeFile(join(consumer, 'runtime.mjs'), `
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { Veya } from 'veya-client';
const cjs = createRequire(import.meta.url)('veya-client');
assert.equal(typeof cjs.Veya, 'function');
const client = new Veya({ fetch: async () => new Response(JSON.stringify({ status: 'ok' })) });
assert.equal((await client.health()).status, 'ok');
assert.equal(typeof client.predictBatch, 'function');
`);
  await exec(process.execPath, ['runtime.mjs'], { cwd: consumer });
  await writeFile(join(consumer, 'types.ts'), `
import { Veya, type Questions, type BatchPrediction } from 'veya-client';
const questions = { q: { type: 'noul', instructions: '?' } } satisfies Questions;
const result: Promise<BatchPrediction<typeof questions>> = new Veya().predictBatch(['state'], questions);
void result;
`);
  await exec(process.execPath, [join(root, 'node_modules/typescript/bin/tsc'), '--noEmit', '--strict',
    '--target', 'ES2022', '--module', 'NodeNext', '--moduleResolution', 'NodeNext', 'types.ts'], { cwd: consumer });
});
