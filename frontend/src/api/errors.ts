/**
 * Frontend-side API error abstraction.
 *
 * UI components should only ever see this shape, never a raw `Response`,
 * raw response text/HTML, or a caught network exception - see
 * `client.ts`'s `apiGet`, which is the only place this is constructed.
 */
export class ApiError extends Error {
  /** HTTP status code, or 0 when the request never reached the server (network failure). */
  readonly status: number;
  /** Backend-provided machine-readable error code, when the backend supplied one. */
  readonly code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}
