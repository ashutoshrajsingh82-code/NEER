// -----------------------------------------------------------------------------
// NEER Frontend — API client  (Phase 33A foundation + Phase 33B endpoints)
//
// The single place the Next.js frontend talks to the NEER FastAPI backend.
// Nothing else in this app should call fetch() against the backend, build a
// backend URL, or parse a backend error body directly — every endpoint
// function below is a thin wrapper over `apiRequest`/`get`/`post`, and every
// component is meant to consume only the normalized `ApiError` shape this
// file produces. Still not called from any page/component yet (Phase 33B
// requirement) — that starts once real pages are built.
//
// Deliberately framework-independent (Requirement 8, Phase 33A): no
// Next.js-specific imports, no "use client" directive, no dependency on
// React. It only relies on the global fetch/AbortController/
// URLSearchParams/FormData, all available in both the browser and Next's
// server/edge runtimes, so this same module works unchanged from Server
// Components, Route Handlers, or client code.
//
// Error normalization is deliberately shaped around the backend's own
// contract (see backend/app/errors.py — every NeerApiError.body() returns
// `{ error: <code>, detail: <message>, ...extra }`) so a component can
// branch on the *same* stable error codes the backend defines (e.g.
// "date_not_found", "model_unavailable") via ApiError.code, without
// re-deriving them here.
//
// Phase 33B's endpoint functions (health, modelInfo, dates, reconstruct,
// reconstructGrid, profile) were written by reading the actual FastAPI
// routers and Pydantic schemas — backend/app/routers/{health,model_info,
// dates,reconstruct,profile}.py and backend/app/schemas.py — rather than
// assumed, so every parameter name, type, and requiredness below matches
// what the backend actually accepts today. No other backend routes
// (embedding, metrics, evaluation, explainability, data_quality,
// reconstruct_netcdf) are wrapped yet; those belong to later phases.
// -----------------------------------------------------------------------------

/**
 * Default request timeout. Generous enough for slower calls later phases
 * will add (e.g. model inference), but still short enough that a hung
 * connection surfaces as a normalized error instead of hanging the UI
 * forever. Pass `timeoutMs: null` per-call to disable it — e.g. when the
 * caller supplies its own long-lived AbortSignal.
 */
const DEFAULT_TIMEOUT_MS = 15_000;

/**
 * Stable, machine-readable codes for failures this module produces itself,
 * as opposed to codes like "date_not_found" that come straight from the
 * backend's own `error` field and are passed through on ApiError.code
 * unchanged.
 */
export const API_ERROR_CODES = {
  CONFIG: "config_error",
  NETWORK: "network_error",
  TIMEOUT: "timeout",
  ABORTED: "aborted",
  INVALID_RESPONSE: "invalid_response",
  VALIDATION: "validation_error",
  HTTP: "http_error",
};

/**
 * Normalized shape every failure from this module takes (Requirement 4) —
 * components never need to branch on whether a failure was a network drop,
 * a timeout, a malformed body, or a real backend error response; they can
 * always read `.code`, `.message`, `.status`, and `.details`.
 */
export class ApiError extends Error {
  /**
   * @param {object} shape
   * @param {string} shape.message - human-readable; safe to show or log directly
   * @param {string} shape.code - one of API_ERROR_CODES, or a backend
   *   error_code (backend/app/errors.py) passed through as-is
   * @param {number|null} [shape.status] - HTTP status, or null when the
   *   request never got a response (network error, timeout, bad config)
   * @param {*} [shape.details] - extra machine-readable context: the
   *   backend's `extra` fields, the raw FastAPI validation-error array, or null
   */
  constructor({ message, code, status = null, details = null }) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

/**
 * Reads and validates the backend origin (Requirement 1). Called lazily,
 * once per request, rather than at module load — importing this file must
 * never crash a build/test that hasn't set the env var yet; the error only
 * surfaces once something actually tries to make a request.
 */
function getBaseUrl() {
  const raw = process.env.NEXT_PUBLIC_API_URL;
  if (!raw || !raw.trim()) {
    throw new ApiError({
      code: API_ERROR_CODES.CONFIG,
      message:
        "NEXT_PUBLIC_API_URL is not set. Configure it in the environment (e.g. .env.local) " +
        "to the NEER FastAPI backend's origin, e.g. http://localhost:8000.",
    });
  }
  // Strip trailing slash(es) so joinUrl never has to worry about doubling up.
  return raw.trim().replace(/\/+$/, "");
}

/** Joins a base origin and a path into one URL with exactly one slash between them. */
function joinUrl(base, path) {
  const cleanPath = `/${String(path).replace(/^\/+/, "")}`;
  return `${base}${cleanPath}`;
}

/**
 * Appends `params` onto `url` as a query string (Requirements 3/6). Every
 * value is encoded via URLSearchParams — never hand-built string
 * concatenation — so arbitrary characters are always safely
 * percent-encoded. `null`/`undefined` values (and array elements) are
 * skipped entirely rather than serialized as the literal string
 * "null"/"undefined"; arrays repeat the key once per element
 * (`tag=a&tag=b`), the common REST convention for multi-value params.
 */
function withQuery(url, params) {
  if (!params) return url;
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) continue;
    if (Array.isArray(value)) {
      for (const item of value) {
        if (item === undefined || item === null) continue;
        search.append(key, String(item));
      }
    } else {
      search.append(key, String(value));
    }
  }
  const query = search.toString();
  return query ? `${url}?${query}` : url;
}

/**
 * Reads and parses a fetch Response's body, tolerating the two "malformed
 * response" cases a real backend can hand back (Requirement 3): no body at
 * all (204/205, or an empty 200), and a body that claims to be JSON but
 * isn't actually parseable. Only throws on a genuinely unparseable
 * non-empty body — the caller turns that into an INVALID_RESPONSE ApiError,
 * preserving the raw text for debugging.
 */
async function parseBody(response) {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    const error = new Error("invalid_json");
    error.rawText = text;
    throw error;
  }
}

/**
 * Turns a non-ok HTTP response into an ApiError, preferring the backend's
 * own `{ error, detail, ...extra }` shape (backend/app/errors.py) when
 * present, so components can branch on the exact same error_code the
 * backend defines. Falls back to FastAPI's default request-validation shape
 * (`{ detail: [...] }`), then a plain `{ detail: "..." }`, then a fully
 * generic HTTP error for anything else.
 */
function errorFromResponse(response, body) {
  if (body && typeof body === "object" && !Array.isArray(body)) {
    if (typeof body.error === "string") {
      const { error, detail, ...extra } = body;
      return new ApiError({
        code: error,
        message: typeof detail === "string" ? detail : response.statusText || "Request failed",
        status: response.status,
        details: Object.keys(extra).length ? extra : null,
      });
    }
    if (Array.isArray(body.detail)) {
      return new ApiError({
        code: API_ERROR_CODES.VALIDATION,
        message: "The request was rejected as invalid.",
        status: response.status,
        details: body.detail,
      });
    }
    if (typeof body.detail === "string") {
      return new ApiError({
        code: API_ERROR_CODES.HTTP,
        message: body.detail,
        status: response.status,
        details: null,
      });
    }
  }
  return new ApiError({
    code: API_ERROR_CODES.HTTP,
    message: response.statusText || `Request failed with status ${response.status}`,
    status: response.status,
    details: body ?? null,
  });
}

/**
 * The one place a fetch() call to the backend is actually made (Requirement
 * 7) — every endpoint function later phases add is meant to be a thin
 * wrapper over this, so HTTP mechanics, query encoding, and error handling
 * live in exactly one place.
 *
 * @param {string} path - backend path, e.g. "/dates" (leading slash optional)
 * @param {object} [options]
 * @param {"GET"|"POST"|"PUT"|"PATCH"|"DELETE"} [options.method="GET"]
 * @param {Object<string, *>} [options.params] - query parameters (any method)
 * @param {*} [options.body] - request body; plain objects/arrays are
 *   JSON-encoded automatically (Content-Type set to application/json);
 *   strings and FormData are sent through as-is
 * @param {Object<string, string>} [options.headers] - merged over the defaults
 * @param {AbortSignal} [options.signal] - caller-supplied cancellation; when
 *   provided, the internal timeout below is skipped — the caller owns
 *   cancellation semantics
 * @param {number|null} [options.timeoutMs] - aborts the request after this
 *   many ms when no `signal` was supplied (default DEFAULT_TIMEOUT_MS);
 *   pass `null` to disable
 * @returns {Promise<*>} the parsed JSON body, or `null` for an empty body
 * @throws {ApiError} always — every failure mode (config, network, timeout,
 *   abort, malformed body, HTTP error) is normalized before it reaches the caller
 */
export async function apiRequest(
  path,
  { method = "GET", params, body, headers, signal, timeoutMs = DEFAULT_TIMEOUT_MS } = {}
) {
  const url = withQuery(joinUrl(getBaseUrl(), path), params);

  const isPreEncodedBody = typeof body === "string" || body instanceof FormData;
  const requestHeaders = { Accept: "application/json", ...headers };
  let requestBody = body;
  if (body !== undefined && !isPreEncodedBody) {
    requestHeaders["Content-Type"] = requestHeaders["Content-Type"] || "application/json";
    requestBody = JSON.stringify(body);
  }

  // Only race our own timeout when the caller hasn't taken over cancellation
  // themselves — composing two independent AbortSignals correctly is more
  // machinery than this foundation needs yet.
  const internalController = !signal && timeoutMs ? new AbortController() : null;
  const timer = internalController ? setTimeout(() => internalController.abort(), timeoutMs) : null;

  let response;
  try {
    response = await fetch(url, {
      method,
      headers: requestHeaders,
      body: requestBody,
      signal: signal ?? internalController?.signal,
    });
  } catch (cause) {
    if (cause?.name === "AbortError") {
      throw new ApiError(
        internalController
          ? { code: API_ERROR_CODES.TIMEOUT, message: `Request to ${path} timed out after ${timeoutMs}ms.` }
          : { code: API_ERROR_CODES.ABORTED, message: `Request to ${path} was aborted.` }
      );
    }
    // fetch() rejects with a generic TypeError for DNS failures, connection
    // refused, CORS blocks, offline, etc. — the Fetch API doesn't expose
    // anything more specific than that to distinguish between them.
    throw new ApiError({
      code: API_ERROR_CODES.NETWORK,
      message: `Could not reach the NEER backend for ${path}. Check your connection and that the backend is running.`,
      details: { cause: cause?.message ?? String(cause) },
    });
  } finally {
    if (timer) clearTimeout(timer);
  }

  let parsedBody;
  try {
    parsedBody = await parseBody(response);
  } catch (parseError) {
    throw new ApiError({
      code: API_ERROR_CODES.INVALID_RESPONSE,
      message: "The server returned a response that could not be parsed as JSON.",
      status: response.status,
      details: { rawText: parseError.rawText },
    });
  }

  if (!response.ok) {
    throw errorFromResponse(response, parsedBody);
  }

  return parsedBody;
}

/**
 * GET convenience wrapper (Requirement 5) — this phase's only supported
 * verb. Query parameters go through the same safe encoding as every other
 * request (see `withQuery`), regardless of which future endpoint function
 * calls this.
 *
 * @param {string} path
 * @param {Object<string, *>} [params]
 * @param {object} [options] - anything else `apiRequest` accepts (headers, signal, timeoutMs)
 */
export function get(path, params, options = {}) {
  return apiRequest(path, { ...options, method: "GET", params });
}

/**
 * POST convenience wrapper. No endpoint function calls this yet (Phase 33A
 * adds none), but `apiRequest` already fully supports a JSON body — this
 * exists purely so a future POST endpoint function is one line built on the
 * same shared request path, instead of a second copy of the fetch/
 * error-handling logic above (Requirement 7).
 *
 * @param {string} path
 * @param {*} [body]
 * @param {object} [options]
 */
export function post(path, body, options = {}) {
  return apiRequest(path, { ...options, method: "POST", body });
}

// -----------------------------------------------------------------------------
// Core NEER endpoints  (Phase 33B)
//
// Every function here is a thin wrapper over `get()` — a query-parameter
// object in, the backend's already-parsed JSON body out, any failure
// normalized to `ApiError` by `apiRequest`. None of them touch fetch, build
// a URL, or handle errors themselves (Requirement: use the centralized
// request helper / do not duplicate fetch logic).
//
// Each accepts an optional trailing `options` object forwarded straight to
// `apiRequest` (currently `signal`/`timeoutMs`/extra `headers`), so a caller
// can cancel or retime any individual call without this file needing a
// bespoke option for it.
// -----------------------------------------------------------------------------

/**
 * Formats a date parameter the way every backend query-param schema expects
 * it (`datetime.date` from a `YYYY-MM-DD` string — see e.g.
 * `PointQueryParams.date` in backend/app/schemas.py). Accepts a `Date`
 * instance for caller convenience and formats it in UTC; a string is passed
 * through untouched (including `undefined`/`null`, so an omitted required
 * date still reaches the backend as a normal missing-param 422 rather than
 * throwing here).
 * @param {string|Date|undefined|null} date
 * @returns {string|undefined|null}
 */
function toDateParam(date) {
  return date instanceof Date ? date.toISOString().slice(0, 10) : date;
}

/**
 * GET /health (backend/app/routers/health.py) — liveness + per-component
 * readiness. Always resolves with `200`; a component being unavailable
 * shows up as `status: "degraded"` in the body, not as a thrown ApiError.
 * @param {object} [options] - forwarded to apiRequest (signal, timeoutMs, headers)
 * @returns {Promise<{status: string, components: Object<string, {status: string, detail?: string}>}>}
 */
export function health(options = {}) {
  return get("/health", undefined, options);
}

/**
 * GET /model/info (backend/app/routers/model_info.py) — architecture,
 * runtime, and checkpoint metadata for whichever model is actually loaded.
 * @param {object} [options]
 * @returns {Promise<{architecture: object, runtime: object, checkpoint: object, environment: string|null}>}
 * @throws {ApiError} code "model_unavailable" (503) — no checkpoint is loaded
 */
export function modelInfo(options = {}) {
  return get("/model/info", undefined, options);
}

/**
 * GET /dates (backend/app/routers/dates.py) — every date actually present
 * in the loaded tensor bundle (never a fabricated/hard-coded range).
 * @param {object} [options]
 * @returns {Promise<{dates: string[], count: number, min_date: string|null, max_date: string|null, data_mode: string, is_synthetic: boolean}>}
 * @throws {ApiError} code "data_unavailable" (503) — no dataset is loaded
 */
export function dates(options = {}) {
  return get("/dates", undefined, options);
}

/**
 * GET /reconstruct (backend/app/routers/reconstruct.py) — reconstructed
 * temperature at one lat/lon/date/depth. All four parameters are required
 * by the backend (`PointQueryParams`); an in-range-but-out-of-NEER-domain
 * coordinate or a `date` absent from the dataset comes back as a normalized
 * `ApiError`, not a thrown exception from here.
 *
 * @param {object} params
 * @param {number} params.lat - degrees, backend-validated to [-90, 90]
 * @param {number} params.lon - degrees, backend-validated to [-180, 360]
 * @param {string|Date} params.date - "YYYY-MM-DD", or a Date (formatted in UTC)
 * @param {number} params.depth - metres, >= 0; snapped to the nearest model depth level
 * @param {object} [options]
 * @returns {Promise<{mode: string, lat: number, lon: number, date: string, depth: number,
 *   temperature: number, anomaly: number, climatology: number|null, embedding_dim: number,
 *   data_mode: string, latency_ms: number, cache_hit: boolean, notes: string[]}>}
 * @throws {ApiError} "invalid_parameter" (400), "date_not_found" (404),
 *   "model_unavailable"/"data_unavailable" (503), "inference_failed" (500),
 *   or "validation_error" (422) for a missing/malformed parameter
 */
export function reconstruct({ lat, lon, date, depth } = {}, options = {}) {
  return get("/reconstruct", { lat, lon, date: toDateParam(date), depth }, options);
}

/**
 * GET /reconstruct/grid (backend/app/routers/reconstruct.py) — reconstructed
 * temperature over a lat/lon region for one date. `depth` is the backend's
 * only optional parameter here (`GridQueryParams.depth`): omit it (leave
 * `undefined`/`null`) to get every model depth level back
 * (`temperature`/`climatology` shaped `(n_lat, n_lon, num_depths)` and
 * `depths` populated); pass one to get a single `(n_lat, n_lon)` slice
 * (`depth` populated instead).
 *
 * @param {object} params
 * @param {number} params.latMin - southern bound, degrees [-90, 90]
 * @param {number} params.latMax - northern bound, degrees [-90, 90]
 * @param {number} params.lonMin - western bound, degrees [-180, 360]
 * @param {number} params.lonMax - eastern bound, degrees [-180, 360]
 * @param {string|Date} params.date - "YYYY-MM-DD", or a Date (formatted in UTC)
 * @param {number} [params.depth] - metres, >= 0; omit for every model depth level
 * @param {object} [options]
 * @returns {Promise<{mode: string, date: string, depth: number|null, depths: number[]|null,
 *   lat: number[], lon: number[], temperature: (number[][]|number[][][]),
 *   climatology: (number[][]|number[][][]|null), data_mode: string, latency_ms: number,
 *   cache_hit: boolean, notes: string[]}>}
 * @throws {ApiError} "invalid_parameter" (400) — e.g. latMin >= latMax, or the
 *   region exceeds the configured cell limit; "grid_unavailable"/"date_not_found"
 *   (404); "model_unavailable"/"data_unavailable" (503); "inference_failed" (500)
 */
export function reconstructGrid({ latMin, latMax, lonMin, lonMax, date, depth } = {}, options = {}) {
  return get(
    "/reconstruct/grid",
    { lat_min: latMin, lat_max: latMax, lon_min: lonMin, lon_max: lonMax, date: toDateParam(date), depth },
    options
  );
}

/**
 * GET /profile (backend/app/routers/profile.py) — the full depth-temperature
 * profile at one lat/lon/date, every model depth level (never interpolated
 * or fabricated). All three parameters are required by the backend
 * (`ProfileQueryParams`).
 *
 * @param {object} params
 * @param {number} params.lat - degrees, backend-validated to [-90, 90]
 * @param {number} params.lon - degrees, backend-validated to [-180, 360]
 * @param {string|Date} params.date - "YYYY-MM-DD", or a Date (formatted in UTC)
 * @param {object} [options]
 * @returns {Promise<{mode: string, lat: number, lon: number, date: string,
 *   depths: number[], temperature: number[], anomaly: number[], climatology: number[]|null,
 *   embedding_dim: number, data_mode: string, latency_ms: number, cache_hit: boolean, notes: string[]}>}
 * @throws {ApiError} "invalid_parameter" (400), "date_not_found" (404),
 *   "model_unavailable"/"data_unavailable" (503), "inference_failed" (500),
 *   or "validation_error" (422) for a missing/malformed parameter
 */
export function profile({ lat, lon, date } = {}, options = {}) {
  return get("/profile", { lat, lon, date: toDateParam(date) }, options);
}