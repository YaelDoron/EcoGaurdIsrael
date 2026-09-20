/**
 * Minimal shared HTTP client for the API layer.
 *
 * `apiGet`/`apiPost` prepend the configured base URL, issue a request,
 * parse JSON, and turn a non-2xx response into an `ApiError`. They
 * deliberately know nothing about any specific endpoint/feature (no
 * FireEvent awareness - see Task 16), do not retry, cache, poll, or attach
 * authentication. Those remain feature/foundation decisions for later
 * tasks to add explicitly, not implicit behavior baked in here.
 *
 * Both accept an optional `signal` (Task A7) so a caller - e.g. a polling
 * hook - can cancel an in-flight request (component unmount, a newer poll
 * superseding a stale one). An aborted request rejects with the browser's
 * own `AbortError` (a `DOMException`), not an `ApiError` - callers that
 * poll are expected to check `error.name === "AbortError"` and treat it as
 * "no update", not a failure to surface.
 *
 * No client-side request timeout is applied here. The Operations Overview
 * endpoint (Task A6) was measured at 3.1-4.6s against the real demo
 * database - an aggressive timeout would misclassify normal responses as
 * failures. Callers that need a bound should pass their own `signal` (e.g.
 * `AbortSignal.timeout(...)`) rather than this module guessing one.
 */
import { getApiBaseUrl } from "./config";
import { ApiError } from "./errors";

const GENERIC_ERROR_MESSAGE = "Something went wrong while talking to the server.";

export interface ApiRequestOptions {
  /** Propagated straight to `fetch` - lets a caller cancel this request (see module docstring). */
  signal?: AbortSignal;
}

interface BackendErrorBody {
  error?: {
    code?: unknown;
    message?: unknown;
  };
}

function isBackendErrorBody(value: unknown): value is BackendErrorBody {
  return typeof value === "object" && value !== null && "error" in value;
}

export async function apiGet<T>(path: string, options: ApiRequestOptions = {}): Promise<T> {
  const url = `${getApiBaseUrl()}${path}`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: options.signal,
    });
  } catch (error) {
    throwIfAborted(error);
    // Network failure, CORS rejection, DNS failure, etc. - never surface
    // the raw browser/fetch error message to the UI.
    throw new ApiError(GENERIC_ERROR_MESSAGE, 0);
  }

  if (!response.ok) {
    throw await toApiError(response);
  }

  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError(GENERIC_ERROR_MESSAGE, response.status);
  }
}

/**
 * POST a JSON body and parse a JSON response. A 202 (e.g. `POST
 * /api/v1/simulation/runs` - Task A3) is a success like any other 2xx; this
 * function never waits beyond the HTTP response itself, so it never blocks
 * on background work the backend performs after responding.
 */
export async function apiPost<T>(path: string, body: unknown, options: ApiRequestOptions = {}): Promise<T> {
  const url = `${getApiBaseUrl()}${path}`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: options.signal,
    });
  } catch (error) {
    throwIfAborted(error);
    throw new ApiError(GENERIC_ERROR_MESSAGE, 0);
  }

  if (!response.ok) {
    throw await toApiError(response);
  }

  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError(GENERIC_ERROR_MESSAGE, response.status);
  }
}

/** Re-throws an intentional AbortController cancellation unchanged - it is not a network failure. */
function throwIfAborted(error: unknown): void {
  if (error instanceof DOMException && error.name === "AbortError") {
    throw error;
  }
}

async function toApiError(response: Response): Promise<ApiError> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    // Non-JSON (e.g. an HTML error page) - never forward raw response text.
    return new ApiError(GENERIC_ERROR_MESSAGE, response.status);
  }

  if (isBackendErrorBody(body) && typeof body.error?.message === "string") {
    const code = typeof body.error.code === "string" ? body.error.code : undefined;
    return new ApiError(body.error.message, response.status, code);
  }

  return new ApiError(GENERIC_ERROR_MESSAGE, response.status);
}
