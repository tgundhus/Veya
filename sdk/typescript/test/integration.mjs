import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';
import { test } from 'node:test';
import { Veya, VeyaAPIError, triageQuestions, emailQuestions, guardQuestions, moderationQuestions, routerQuestions } from 'veya-client';

test('JavaScript → HTTP → Python Router → real offline Agent inference', { timeout: 90_000 }, async t => {
  const script = fileURLToPath(new URL('../../../tests/sdk_server_fixture.py', import.meta.url));
  const server = spawn(process.env.PYTHON ?? 'python3', [script], { stdio: ['ignore', 'pipe', 'pipe'] });
  let stderr = '';
  server.stderr.on('data', chunk => { stderr += chunk; });
  t.after(async () => {
    if (server.exitCode === null && server.signalCode === null) {
      const exited = once(server, 'exit');
      server.kill('SIGTERM');
      const timer = setTimeout(() => server.kill('SIGKILL'), 5000);
      try { await exited; } finally { clearTimeout(timer); }
    }
  });
  const fixture = await new Promise((resolve, reject) => {
    const lines = createInterface({ input: server.stdout });
    const timer = setTimeout(() => reject(new Error(`Server startup timed out: ${stderr}`)), 60_000);
    const failed = code => { clearTimeout(timer); reject(new Error(`Server exited (${code}): ${stderr}`)); };
    server.once('exit', failed);
    server.once('error', error => { clearTimeout(timer); reject(error); });
    lines.on('line', line => {
      if (line.startsWith('VEYA_TEST_SERVER=')) {
        clearTimeout(timer);
        server.off('exit', failed);
        resolve(JSON.parse(line.slice('VEYA_TEST_SERVER='.length)));
      }
    });
  });
  const client = new Veya({ baseURL: fixture.baseURL, apiKey: 'integration-test', timeoutMs: 10_000 });
  let ready = false;
  for (let i = 0; i < 100; i++) {
    try { await client.health(); ready = true; break; } catch { await delay(50); }
  }
  assert.ok(ready, `Server never became ready: ${stderr}`);

  const result = await client.predict(fixture.state, fixture.questions, { model: 'english' });
  const { routing, ...prediction } = result;
  const modelAnswers = Object.fromEntries(Object.entries(prediction.answers).map(([id, answer]) => {
    const { x_jev_confidence: compatibilityConfidence, ...modelAnswer } = answer;
    if (answer.type === 'choice' || answer.type === 'score') {
      assert.equal(typeof compatibilityConfidence, 'number', `${id} must include HTTP compatibility confidence`);
      assert.ok(Number.isFinite(compatibilityConfidence) && compatibilityConfidence >= 0 && compatibilityConfidence <= 1);
    } else {
      assert.equal(compatibilityConfidence, undefined, 'noul has no compatibility confidence extension');
    }
    return [id, modelAnswer];
  }));
  assert.deepEqual({ ...prediction, answers: modelAnswers }, fixture.expected,
    'SDK model answers must exactly match direct Python inference');
  assert.equal(routing.model, 'english');
  assert.equal(result.answers.single.choice, 'only');
  assert.equal(result.answers.single.probabilities.only, 1);
  assert.equal(result.answers.single.x_jev_confidence, 1);
  assert.ok(result.usage.input_tokens > 0);
  assert.equal(result.usage.output_tokens, 0);
  const batch = await client.predictBatch([fixture.state, fixture.state], fixture.questions,
    { model: 'english', batchSize: 1 });
  assert.equal(batch.results.length, 2);
  assert.deepEqual(batch.results[0].answers, result.answers);
  assert.deepEqual(batch.results[1].answers, result.answers);
  assert.equal(batch.total_usage.output_tokens, 0);
  assert.equal((await client.predict({ text: 'मुझे पैसे वापस चाहिए' }, fixture.questions)).routing.model, 'multilingual');
  assert.equal((await client.predict('hello', fixture.questions, { model: 'typed_decisions' })).routing.model, 'typed-decisions');

  for (const preset of [triageQuestions, emailQuestions, guardQuestions, moderationQuestions, routerQuestions]) {
    const questions = preset();
    const response = await client.predict({ message: 'hello', body: 'hello', prompt: 'hello', post: 'hello', request: 'hello' }, questions);
    assert.deepEqual(Object.keys(response.answers), Object.keys(questions));
  }
  const unauthenticated = new Veya({ baseURL: fixture.baseURL });
  assert.equal((await unauthenticated.health()).status, 'ok');
  await assert.rejects(unauthenticated.predict('hello', fixture.questions), error =>
    error instanceof VeyaAPIError && error.status === 401 && error.message === 'invalid or missing bearer token');
  await assert.rejects(client.predict('x'.repeat(50_001), fixture.questions), error =>
    error instanceof VeyaAPIError && error.status === 413 && error.message.includes('state too large'));
  assert.equal((await fetch(`${fixture.baseURL}/v1/predict`, { method: 'POST' })).status, 404);
  assert.equal((await fetch(`${fixture.baseURL}/v1/route`, { method: 'POST' })).status, 404);
});
