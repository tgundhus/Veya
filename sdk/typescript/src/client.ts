import {
  VeyaAbortError, VeyaAPIError, VeyaConnectionError, VeyaError, VeyaResponseError,
  VeyaTimeoutError, VeyaValidationError,
} from './errors.js';
import type { BatchPrediction, Health, Prediction, PredictBatchOptions, PredictOptions, Questions, RequestOptions, State } from './types.js';
import { isMinConfidenceKey } from './buckets.js';
import { isRecord, validateJson, validateQuestions, validateTimeout } from './validation.js';
import { validateHealth, validatePrediction } from './response.js';

/** One threshold in `[0, 1]`: booleans are not thresholds even though they read as 0/1. */
function isThreshold(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;
}

function validateState(state: unknown): void {
  if (typeof state !== 'string' && !Array.isArray(state) && !isRecord(state)) {
    throw new VeyaValidationError('state must be text, a JSON object, or an array');
  }
  validateJson(state);
}

/** Mirror core's `check_min_confidence`: one threshold for every answer, or a non-empty
 *  per-bucket map whose keys core can produce and whose values are all thresholds. Anything else
 *  is refused before the request goes out, so a caller cannot pay for inference with a gate the
 *  server would 422. */
function validateMinConfidence(value: unknown): asserts value is number | Record<string, number> {
  if (isThreshold(value)) return;
  if (isRecord(value) && Object.keys(value).length > 0
      && Object.keys(value).every(isMinConfidenceKey)
      && Object.values(value).every(isThreshold)) return;
  throw new VeyaValidationError(
    'minConfidence must be a number between 0 and 1, or a non-empty map of bucket to threshold');
}

export interface VeyaOptions {
  /** Server root URL, optionally including a reverse-proxy path prefix. */
  baseURL?: string;
  apiKey?: string;
  /** Optional local checkpoint or alias to send with every prediction. */
  model?: string;
  /** Default 120000 ms; cold checkpoint downloads may require more. Zero disables. */
  timeoutMs?: number;
  headers?: HeadersInit;
  /** Supply a Fetch-compatible implementation for tests or a custom transport. */
  fetch?: typeof globalThis.fetch;
}

export class Veya {
  private readonly baseURL: string;
  private readonly timeoutMs: number;
  private readonly headers: Headers;
  private readonly fetcher: typeof globalThis.fetch;
  private readonly model: string | undefined;

  constructor(options: VeyaOptions = {}) {
    let url: URL;
    try { url = new URL(options.baseURL ?? 'http://127.0.0.1:8000'); }
    catch { throw new VeyaValidationError('baseURL must be an absolute HTTP(S) URL'); }
    if (!['http:', 'https:'].includes(url.protocol) || url.search || url.hash || url.username || url.password) {
      throw new VeyaValidationError('baseURL must be HTTP(S), without credentials, query, or fragment');
    }
    this.baseURL = url.href.replace(/\/+$/, '');
    this.model = options.model;
    if (this.model !== undefined && (typeof this.model !== 'string' || !this.model.trim())) {
      throw new VeyaValidationError('model must be a nonempty string');
    }
    this.timeoutMs = options.timeoutMs ?? 120_000;
    validateTimeout(this.timeoutMs);
    this.headers = new Headers(options.headers);
    this.headers.set('Accept', 'application/json');
    if (options.apiKey !== undefined) this.headers.set('Authorization', `Bearer ${options.apiKey}`);
    const fetcher = options.fetch ?? globalThis.fetch;
    if (typeof fetcher !== 'function') throw new VeyaValidationError('A Fetch implementation is required');
    this.fetcher = fetcher;
  }

  async predict<const Q extends Questions>(state: State, questions: Q, options: PredictOptions = {}): Promise<Prediction<Q>> {
    validateQuestions(questions);
    const result = await this.request<Prediction<Q>>('/v1/systemone', this.body(state, questions, options), options);
    validatePrediction(result, questions);
    return result;
  }

  /** Answer states sharing one question set in input order through the server's batch endpoint. */
  async predictBatch<const Q extends Questions>(states: readonly State[], questions: Q,
    options: PredictBatchOptions = {}): Promise<BatchPrediction<Q>> {
    if (!Array.isArray(states) || states.length === 0) throw new VeyaValidationError('states must be a nonempty array');
    validateQuestions(questions);
    for (const state of states) validateState(state);
    const body = JSON.parse(this.body(states[0], questions, options));
    delete body.state;
    body.states = states;
    if (options.batchSize !== undefined) {
      if (!Number.isInteger(options.batchSize) || options.batchSize < 1) {
        throw new VeyaValidationError('batchSize must be a positive integer');
      }
      body.batch_size = options.batchSize;
    }
    if (options.sortByLength !== undefined) {
      if (typeof options.sortByLength !== 'boolean') throw new VeyaValidationError('sortByLength must be boolean');
      body.sort_by_length = options.sortByLength;
    }
    const result = await this.request<BatchPrediction<Q>>('/v1/systemone/batch', JSON.stringify(body), options);
    if (!Array.isArray(result.results) || result.results.length !== states.length ||
        !isRecord(result.total_usage) || !Number.isInteger(result.total_usage.input_tokens) ||
        !Number.isInteger(result.total_usage.output_tokens)) {
      throw new VeyaResponseError('Invalid Veya batch response');
    }
    for (const prediction of result.results) validatePrediction(prediction, questions);
    return result;
  }

  /** Query the self-hosted Veya server's health probe. */
  async health(options: RequestOptions = {}): Promise<Health> {
    const result = await this.request<Health>('/health', undefined, options);
    validateHealth(result);
    return result;
  }

  private body(state: State, questions: Questions, options: PredictOptions): string {
    validateState(state);
    if (options.task !== undefined && (typeof options.task !== 'string' || !options.task.trim())) {
      throw new VeyaValidationError('task must be a nonempty string');
    }
    for (const key of ['lang', 'langGuess'] as const) {
      const value = options[key];
      if (value !== undefined && (typeof value !== 'string' || !value.trim())) {
        throw new VeyaValidationError(`${key} must be a nonempty language code string`);
      }
    }
    for (const key of ['maxLen', 'headMaxLen'] as const) {
      const value = options[key];
      if (value !== undefined && (!Number.isInteger(value) || value < 1)) {
        throw new VeyaValidationError(`${key} must be a positive integer`);
      }
    }
    if (options.minConfidence !== undefined) validateMinConfidence(options.minConfidence);
    const model = options.model ?? this.model;
    if (model !== undefined && (typeof model !== 'string' || !model.trim())) {
      throw new VeyaValidationError('model must be a nonempty string');
    }
    const wireQuestions = Object.fromEntries(Object.entries(questions).map(([id, question]) => [id,
      question.type === 'choice' && Array.isArray(question.criteria)
        ? { ...question, criteria: Object.fromEntries(question.criteria.map(label => [label, null])) }
        : question,
    ]));
    const body: {
      state: State; questions: typeof wireQuestions; model?: string; task?: string;
      lang?: string; lang_guess?: string; max_len?: number; head_max_len?: number;
      min_confidence?: number | Record<string, number>;
    } = { state, questions: wireQuestions };
    if (model !== undefined) body.model = model;
    // The per-request controls `/v1/systemone` forwards when the request sends them (see
    // docs/http-api.md); absent options stay absent so the deployment's own Router
    // settings remain in charge.
    if (options.task !== undefined) body.task = options.task;
    if (options.lang !== undefined) body.lang = options.lang;
    if (options.langGuess !== undefined) body.lang_guess = options.langGuess;
    if (options.maxLen !== undefined) body.max_len = options.maxLen;
    if (options.headMaxLen !== undefined) body.head_max_len = options.headMaxLen;
    if (options.minConfidence !== undefined) body.min_confidence = options.minConfidence;
    validateJson(body);
    return JSON.stringify(body);
  }

  private async request<T>(path: string, body: string | undefined, options: RequestOptions): Promise<T> {
    const timeoutMs = options.timeoutMs ?? this.timeoutMs;
    validateTimeout(timeoutMs);
    const controller = new AbortController();
    const abort = () => controller.abort(new VeyaAbortError('Request aborted', { cause: options.signal?.reason }));
    options.signal?.addEventListener('abort', abort, { once: true });
    if (options.signal?.aborted) abort();
    const timer = timeoutMs > 0 ? setTimeout(() => {
      controller.abort(new VeyaTimeoutError(`Request timed out after ${timeoutMs} ms`));
    }, timeoutMs) : undefined;

    try {
      controller.signal.throwIfAborted();
      const headers = new Headers(this.headers);
      if (body !== undefined) headers.set('Content-Type', 'application/json');
      const init: RequestInit = { method: body === undefined ? 'GET' : 'POST', headers, signal: controller.signal,
        redirect: 'error' };
      if (body !== undefined) init.body = body;
      // Never retry automatically: repeating an expensive inference request is surprising.
      const response = await this.fetcher.call(globalThis, this.baseURL + path, init);
      const raw = await response.text();
      let payload: unknown;
      try { payload = JSON.parse(raw); }
      catch {
        if (response.ok) throw new VeyaResponseError('Server returned invalid JSON');
      }
      if (!response.ok) {
        const error = isRecord(payload) && isRecord(payload.error) ? payload.error : undefined;
        const detail = isRecord(payload) ? payload.detail : undefined;
        throw new VeyaAPIError(
          typeof error?.message === 'string' ? error.message
            : typeof detail === 'string' ? detail : `Veya returned HTTP ${response.status}`,
          response.status,
          typeof error?.code === 'string' ? error.code : 'http_error',
          error?.details ?? detail ?? payload ?? raw,
        );
      }
      if (!isRecord(payload)) throw new VeyaResponseError('Server returned a non-object JSON response');
      return payload as T;
    } catch (error) {
      if (controller.signal.aborted) throw controller.signal.reason;
      if (error instanceof VeyaError) throw error;
      throw new VeyaConnectionError('Could not reach the Veya server', { cause: error });
    } finally {
      if (timer !== undefined) clearTimeout(timer);
      options.signal?.removeEventListener('abort', abort);
    }
  }
}
