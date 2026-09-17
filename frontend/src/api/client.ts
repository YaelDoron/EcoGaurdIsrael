/**
 * Minimal shared HTTP client for the API layer.
 *
 * `apiGet` prepends the configured base URL, issues a GET request, parses
 * JSON, and turns a non-2xx response into an `ApiError`. It deliberately
 * knows nothing about any specific endpoint/feature (no FireEvent
 * awareness - see Task 16), does not retry, cache, poll, or attach
 * authentication. Those remain feature/foundation decisions for later
 * tasks to add explicitly, not implicit behavior baked in here.
 */
import { getApiBaseUrl } from "./config";
import { ApiError } from "./errors";

const GENERIC_ERROR_MESSAGE = "Something went wrong while talking to the server.";

interface BackendErrorBody {
  error?: {
    code?: unknown;
    message?: unknown;
  };
}

function isBackendErrorBody(value: unknown): value is BackendErrorBody {
  return typeof value === "object" && value !== null && "error" in value;
}

export async function apiGet<T>(path: string): Promise<T> {
  const url = `${getApiBaseUrl()}${path}`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
  } catch {
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
