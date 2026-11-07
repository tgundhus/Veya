import { VeyaValidationError } from './errors.js';

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Reject JS values JSON.stringify would silently drop or rewrite. */
export function validateJson(value: unknown, path = 'request', ancestors = new Set<object>()): void {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return;
  if (typeof value === 'number' && Number.isFinite(value)) return;
  if (typeof value !== 'object' || value === null) {
    throw new VeyaValidationError(`${path} must contain only JSON values`);
  }
  if (ancestors.has(value)) throw new VeyaValidationError(`${path} contains a circular reference`);
  if (ancestors.size > 100) throw new VeyaValidationError(`${path} exceeds the JSON nesting limit (100)`);
  if (!Array.isArray(value) && Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null) {
    throw new VeyaValidationError(`${path} must be a plain JSON object`);
  }
  ancestors.add(value);
  if (Array.isArray(value)) {
    for (const item of value) validateJson(item, `${path}[]`, ancestors);
  } else {
    for (const [key, item] of Object.entries(value)) validateJson(item, `${path}.${key}`, ancestors);
  }
  ancestors.delete(value);
}

export function validateQuestions(questions: unknown): void {
  if (!isRecord(questions) || Object.keys(questions).length === 0) {
    throw new VeyaValidationError('questions must be an object containing at least one question');
  }
  for (const [id, q] of Object.entries(questions)) {
    const invalid = (message: string): never => { throw new VeyaValidationError(`questions.${id}: ${message}`); };
    if (!isRecord(q)) invalid('expected a question object');
    const question = q as Record<string, unknown>;
    if (!Object.hasOwn(question, 'instructions')) invalid('instructions is required');
    if (Object.keys(question).some(key => !['type', 'instructions', 'criteria', 'labels', 'option_order'].includes(key))) {
      invalid('unknown question field');
    }
    if (Object.hasOwn(question, 'labels') && question.type !== 'noul') invalid('labels require a noul question');
    const criteria = question.criteria;
    switch (question.type) {
      case 'choice':
        if (Array.isArray(criteria)) {
          if (!criteria.length || criteria.some(c => typeof c !== 'string') || new Set(criteria).size !== criteria.length) {
            invalid('choice criteria must contain unique string labels');
          }
        } else if (!isRecord(criteria) || !Object.keys(criteria).length) {
          invalid('choice requires a nonempty criteria object or array');
        }
        break;
      case 'score':
        if (!Array.isArray(criteria) || !criteria.length) invalid('score requires a nonempty criteria array');
        if ((criteria as unknown[]).some(level => level === null || level === undefined)) {
          invalid('score criteria cannot contain null levels');
        }
        break;
      case 'noul':
        if (criteria !== undefined && criteria !== null &&
            (!isRecord(criteria) || Object.keys(criteria).some(k => k !== 'true' && k !== 'false'))) {
          invalid('noul criteria may only describe true and false');
        }
        if (question.labels !== undefined && question.labels !== null) {
          const labels = question.labels;
          if (!isRecord(labels) || Object.keys(labels).length !== 2 ||
              typeof labels.false !== 'string' || typeof labels.true !== 'string' ||
              !labels.false.trim() || !labels.true.trim() || labels.false === labels.true) {
            invalid('labels must contain distinct nonempty false and true strings');
          }
        }
        break;
      default: invalid('type must be choice, score, or noul');
    }
    if (question.option_order !== undefined && question.option_order !== null) {
      const count = question.type === 'noul' ? 2 : Array.isArray(criteria) ? criteria.length
        : Object.keys(criteria as object).length;
      const order = question.option_order;
      if (!Array.isArray(order) || order.length !== count || new Set(order).size !== count ||
          order.some(index => !Number.isInteger(index) || index < 0 || index >= count)) {
        invalid('option_order must be a permutation of the option indices');
      }
    }
  }
}

export function validateTimeout(value: number): void {
  if (!Number.isFinite(value) || value < 0 || value > 2_147_483_647) {
    throw new VeyaValidationError('timeoutMs must be between 0 and 2147483647');
  }
}
