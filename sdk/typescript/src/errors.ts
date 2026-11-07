export class VeyaError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = new.target.name;
  }
}

/** Request rejected locally, before any network traffic. */
export class VeyaValidationError extends VeyaError {}

/** A non-2xx response. Includes JSON error details when available. */
export class VeyaAPIError extends VeyaError {
  constructor(
    message: string,
    public readonly status: number,
    public readonly code: string,
    public readonly details: unknown,
  ) {
    super(message);
  }
}

export class VeyaConnectionError extends VeyaError {}
export class VeyaTimeoutError extends VeyaError {}
export class VeyaAbortError extends VeyaError {}
export class VeyaResponseError extends VeyaError {}
