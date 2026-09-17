/**
 * Centralized access to the backend base URL.
 *
 * Policy: if `VITE_API_BASE_URL` is not set, fall back to
 * `http://localhost:8000` (the backend's own default local dev port - see
 * `backend/src/api/app.py`) rather than throwing at startup. This keeps
 * `npm run dev` usable out of the box for a fresh checkout; production/
 * deployed builds are expected to set `VITE_API_BASE_URL` explicitly (see
 * `.env.example`).
 *
 * No component should read `import.meta.env.VITE_API_BASE_URL` directly -
 * always go through `getApiBaseUrl()` so this policy lives in one place.
 */
const DEFAULT_API_BASE_URL = "http://localhost:8000";

export function getApiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL;
  return configured && configured.trim() !== "" ? configured : DEFAULT_API_BASE_URL;
}
