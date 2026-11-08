export { Veya } from './client.js';
export type { VeyaOptions } from './client.js';
export * from './types.js';
export * from './errors.js';
export * from './presets.js';

// Legacy imports share the same constructors.
export { Veya as Laya } from "./client.js";
export type { VeyaOptions as LayaOptions } from "./client.js";
export { VeyaError as LayaError, VeyaValidationError as LayaValidationError, VeyaAPIError as LayaAPIError, VeyaConnectionError as LayaConnectionError, VeyaTimeoutError as LayaTimeoutError, VeyaAbortError as LayaAbortError, VeyaResponseError as LayaResponseError } from "./errors.js";
