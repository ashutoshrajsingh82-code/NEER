// -----------------------------------------------------------------------------
// NEER Frontend API Client
//
// Centralized API communication layer for the NEER frontend.
//
// Responsibilities:
// - API base URL management
// - GET / POST requests
// - timeout and abort handling
// - HTTP error normalization
// - response parsing
// - date normalization
// - backend response validation
// - scientific data endpoints
//
// Phase 37A:
// - runReconstruction()
// -----------------------------------------------------------------------------

import {
  isIsoDate,
  parseDatesPayload,
} from "./dateDepthModel.js";

import {
  validatePointReconstruction,
} from "./reconstructionResponse.js";

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

const DEFAULT_TIMEOUT_MS = 15_000;

export const API_ERROR_CODES = {
  CONFIG: "config_error",
  NETWORK: "network_error",
  TIMEOUT: "timeout",
  ABORTED: "aborted",
  INVALID_RESPONSE: "invalid_response",
  VALIDATION: "validation_error",
  HTTP: "http_error",
};

// -----------------------------------------------------------------------------
// ApiError
// -----------------------------------------------------------------------------

export class ApiError extends Error {
  constructor({
    code,
    message,
    status = null,
    details = null,
    cause = null,
  } = {}) {
    super(message || "API request failed");

    this.name = "ApiError";
    this.code = code || API_ERROR_CODES.NETWORK;
    this.status = status;
    this.details = details;
    this.cause = cause;
  }
}

// -----------------------------------------------------------------------------
// Error categorization
// -----------------------------------------------------------------------------

export function categorizeApiError(error) {
  if (error instanceof ApiError) {
    return error;
  }

  if (error?.name === "AbortError") {
    return new ApiError({
      code: API_ERROR_CODES.ABORTED,
      message: "The API request was aborted.",
      cause: error,
    });
  }

  return new ApiError({
    code: API_ERROR_CODES.NETWORK,
    message: error?.message || "Network request failed.",
    cause: error,
  });
}

// -----------------------------------------------------------------------------
// Base URL
// -----------------------------------------------------------------------------

export function getBaseUrl() {
  const value = process.env.NEXT_PUBLIC_API_URL;

  if (!value || typeof value !== "string") {
    throw new ApiError({
      code: API_ERROR_CODES.CONFIG,
      message:
        "NEXT_PUBLIC_API_URL is not configured. Set it in the frontend environment.",
    });
  }

  return value.replace(/\/+$/, "");
}

// -----------------------------------------------------------------------------
// URL helpers
// -----------------------------------------------------------------------------

function joinUrl(baseUrl, path) {
  const base = baseUrl.replace(/\/+$/, "");
  const cleanPath = String(path || "").replace(/^\/+/, "");

  return `${base}/${cleanPath}`;
}

function withQuery(url, query = {}) {
  const params = new URLSearchParams();

  Object.entries(query).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") {
      return;
    }

    params.set(key, String(value));
  });

  const queryString = params.toString();

  return queryString ? `${url}?${queryString}` : url;
}

// -----------------------------------------------------------------------------
// Response parsing
// -----------------------------------------------------------------------------

async function parseBody(response, responseType = "json") {
  if (responseType === "blob") {
    return response.blob();
  }

  if (responseType === "text") {
    return response.text();
  }

  const text = await response.text();

  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text);
  } catch (error) {
    throw new ApiError({
      code: API_ERROR_CODES.INVALID_RESPONSE,
      message: "The API returned invalid JSON.",
      status: response.status,
      cause: error,
      details: {
        body: text,
      },
    });
  }
}

// -----------------------------------------------------------------------------
// Content-Disposition filename
// -----------------------------------------------------------------------------

function filenameFromContentDisposition(value) {
  if (!value || typeof value !== "string") {
    return null;
  }

  const utf8Match = value.match(
    /filename\*=UTF-8''([^;]+)/i
  );

  if (utf8Match?.[1]) {
    try {
      return decodeURIComponent(utf8Match[1]);
    } catch {
      return utf8Match[1];
    }
  }

  const standardMatch = value.match(
    /filename="?([^";]+)"?/i
  );

  return standardMatch?.[1] || null;
}

// -----------------------------------------------------------------------------
// HTTP error parsing
// -----------------------------------------------------------------------------

async function errorFromResponse(response) {
  let body = null;

  try {
    body = await parseBody(response, "json");
  } catch {
    body = null;
  }

  const detail =
    body?.detail ??
    body?.message ??
    body?.error ??
    null;

  let message;

  if (typeof detail === "string") {
    message = detail;
  } else if (detail && typeof detail === "object") {
    message =
      detail.message ||
      detail.detail ||
      JSON.stringify(detail);
  } else {
    message =
      `API request failed with HTTP ${response.status}.`;
  }

  return new ApiError({
    code: API_ERROR_CODES.HTTP,
    message,
    status: response.status,
    details: body,
  });
}

// -----------------------------------------------------------------------------
// Core API request
// -----------------------------------------------------------------------------

export async function apiRequest(
  path,
  {
    method = "GET",
    query = {},
    body,
    headers = {},
    signal,
    timeoutMs = DEFAULT_TIMEOUT_MS,
    responseType = "json",
  } = {}
) {
  let url;

  try {
    url = withQuery(
      joinUrl(getBaseUrl(), path),
      query
    );
  } catch (error) {
    throw categorizeApiError(error);
  }

  const controller = new AbortController();

  let timedOut = false;

  const timeoutId = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  const forwardAbort = () => {
    controller.abort();
  };

  if (signal) {
    if (signal.aborted) {
      clearTimeout(timeoutId);

      throw new ApiError({
        code: API_ERROR_CODES.ABORTED,
        message: "The API request was aborted.",
      });
    }

    signal.addEventListener(
      "abort",
      forwardAbort,
      { once: true }
    );
  }

  const requestHeaders = {
    Accept: "application/json",
    ...headers,
  };

  const requestOptions = {
    method,
    headers: requestHeaders,
    signal: controller.signal,
  };

  if (body !== undefined) {
    if (
      typeof body === "object" &&
      body !== null &&
      !(body instanceof FormData) &&
      !(body instanceof Blob)
    ) {
      requestHeaders["Content-Type"] =
        requestHeaders["Content-Type"] ||
        "application/json";

      requestOptions.body = JSON.stringify(body);
    } else {
      requestOptions.body = body;
    }
  }

  try {
    const response = await fetch(
      url,
      requestOptions
    );

    if (!response.ok) {
      throw await errorFromResponse(response);
    }

    const parsed = await parseBody(
      response,
      responseType
    );

    return parsed;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }

    if (timedOut) {
      throw new ApiError({
        code: API_ERROR_CODES.TIMEOUT,
        message:
          `The API request timed out after ${timeoutMs} ms.`,
        cause: error,
      });
    }

    if (
      error?.name === "AbortError" ||
      signal?.aborted
    ) {
      throw new ApiError({
        code: API_ERROR_CODES.ABORTED,
        message: "The API request was aborted.",
        cause: error,
      });
    }

    throw new ApiError({
      code: API_ERROR_CODES.NETWORK,
      message:
        error?.message ||
        "Unable to reach the NEER backend.",
      cause: error,
    });
  } finally {
    clearTimeout(timeoutId);

    if (signal) {
      signal.removeEventListener(
        "abort",
        forwardAbort
      );
    }
  }
}

// -----------------------------------------------------------------------------
// Convenience methods
// -----------------------------------------------------------------------------

export function get(
  path,
  query = {},
  options = {}
) {
  return apiRequest(path, {
    ...options,
    method: "GET",
    query,
  });
}

export function post(
  path,
  body,
  options = {}
) {
  return apiRequest(path, {
    ...options,
    method: "POST",
    body,
  });
}

// -----------------------------------------------------------------------------
// Date helper
// -----------------------------------------------------------------------------

export function toDateParam(value) {
  if (value === null || value === undefined) {
    return null;
  }

  if (typeof value === "string") {
    return value;
  }

  if (value instanceof Date) {
    if (Number.isNaN(value.getTime())) {
      return null;
    }

    return value.toISOString().slice(0, 10);
  }

  return String(value);
}

// -----------------------------------------------------------------------------
// Health
// -----------------------------------------------------------------------------
//
// GET /health
// -----------------------------------------------------------------------------

export function health(options = {}) {
  return get("/health", {}, options);
}

// -----------------------------------------------------------------------------
// Model information
// -----------------------------------------------------------------------------
//
// GET /model/info
// -----------------------------------------------------------------------------

export function modelInfo(options = {}) {
  return get("/model/info", {}, options);
}

// -----------------------------------------------------------------------------
// Available dates
// -----------------------------------------------------------------------------
//
// GET /dates
//
// Response is normalized through dateDepthModel.js.
// -----------------------------------------------------------------------------

export async function dates(options = {}) {
  const payload = await get(
    "/dates",
    {},
    options
  );

  return parseDatesPayload(payload);
}

// -----------------------------------------------------------------------------
// Point reconstruction
// -----------------------------------------------------------------------------
//
// GET /reconstruct
//
// Existing low-level endpoint wrapper.
//
// Parameters:
// - lat
// - lon
// - date
// - depth
// -----------------------------------------------------------------------------

export function reconstruct(
  { lat, lon, date, depth } = {},
  options = {}
) {
  return get(
    "/reconstruct",
    {
      lat,
      lon,
      date: toDateParam(date),
      depth,
    },
    options
  );
}

// -----------------------------------------------------------------------------
// Phase 37A — LIVE RECONSTRUCTION
// -----------------------------------------------------------------------------
//
// Executes one real point reconstruction against the backend.
//
// Backend contract:
//
// GET /reconstruct?lat=&lon=&date=&depth=
//
// The function:
// 1. validates date
// 2. validates depth
// 3. validates latitude
// 4. validates longitude
// 5. calls the existing reconstruct() endpoint
// 6. validates the backend response
// 7. returns the validated reconstruction
//
// No mock data.
// No fallback values.
// No fabricated success.
// No swallowed errors.
//
// The backend may snap the requested depth to the nearest model level.
// Therefore the returned `depth` is the actual depth used by the backend.
// -----------------------------------------------------------------------------

export async function runReconstruction(
  {
    date,
    depth,
    latitude,
    longitude,
  } = {},
  options = {}
) {
  const isoDate =
    date instanceof Date
      ? Number.isNaN(date.getTime())
        ? null
        : toDateParam(date)
      : date;

  const problems = [];

  if (!isIsoDate(isoDate)) {
    problems.push(
      'date must be a valid "YYYY-MM-DD" date'
    );
  }

  if (
    typeof depth !== "number" ||
    !Number.isFinite(depth) ||
    depth < 0
  ) {
    problems.push(
      "depth must be a finite number of metres >= 0"
    );
  }

  if (
    typeof latitude !== "number" ||
    !Number.isFinite(latitude) ||
    latitude < -90 ||
    latitude > 90
  ) {
    problems.push(
      "latitude must be a finite number in [-90, 90]"
    );
  }

  if (
    typeof longitude !== "number" ||
    !Number.isFinite(longitude) ||
    longitude < -180 ||
    longitude > 360
  ) {
    problems.push(
      "longitude must be a finite number in [-180, 360]"
    );
  }

  if (problems.length > 0) {
    throw new ApiError({
      code: API_ERROR_CODES.VALIDATION,
      message:
        `Cannot run reconstruction: ${problems.join("; ")}.`,
      details: {
        problems,
      },
    });
  }

  const body = await reconstruct(
    {
      lat: latitude,
      lon: longitude,
      date: isoDate,
      depth,
    },
    options
  );

  const result =
    validatePointReconstruction(
      body,
      {
        date: isoDate,
      }
    );

  if (!result.ok) {
    throw new ApiError({
      code: API_ERROR_CODES.INVALID_RESPONSE,
      message: result.reason,
      status: 200,
      details: {
        fields: result.fields,
        body,
      },
    });
  }

  return result.value;
}

// -----------------------------------------------------------------------------
// Reconstruction grid
// -----------------------------------------------------------------------------
//
// GET /reconstruct/grid
//
// Returns the reconstructed temperature field over the configured region.
// -----------------------------------------------------------------------------

export function reconstructGrid(
  { date, depth } = {},
  options = {}
) {
  return get(
    "/reconstruct/grid",
    {
      date: toDateParam(date),
      depth,
    },
    options
  );
}

// -----------------------------------------------------------------------------
// Profile
// -----------------------------------------------------------------------------
//
// GET /profile
// -----------------------------------------------------------------------------

export function profile(
  { lat, lon, date } = {},
  options = {}
) {
  return get(
    "/profile",
    {
      lat,
      lon,
      date: toDateParam(date),
    },
    options
  );
}

// -----------------------------------------------------------------------------
// Embedding
// -----------------------------------------------------------------------------
//
// GET /embedding
// -----------------------------------------------------------------------------

export function embedding(
  { lat, lon, date } = {},
  options = {}
) {
  return get(
    "/embedding",
    {
      lat,
      lon,
      date: toDateParam(date),
    },
    options
  );
}

// -----------------------------------------------------------------------------
// Metrics
// -----------------------------------------------------------------------------
//
// GET /metrics
// -----------------------------------------------------------------------------

export function metrics(options = {}) {
  return get(
    "/metrics",
    {},
    options
  );
}

// -----------------------------------------------------------------------------
// Argo evaluation
// -----------------------------------------------------------------------------
//
// GET /argo/evaluation
// -----------------------------------------------------------------------------

export function argoEvaluation(
  params = {},
  options = {}
) {
  return get(
    "/argo/evaluation",
    params,
    options
  );
}

// -----------------------------------------------------------------------------
// Explainability
// -----------------------------------------------------------------------------
//
// GET /explainability
// -----------------------------------------------------------------------------

export function explainability(
  params = {},
  options = {}
) {
  return get(
    "/explainability",
    params,
    options
  );
}

// -----------------------------------------------------------------------------
// Data quality
// -----------------------------------------------------------------------------
//
// GET /data/quality
// -----------------------------------------------------------------------------

export function dataQuality(
  params = {},
  options = {}
) {
  return get(
    "/data/quality",
    params,
    options
  );
}

// -----------------------------------------------------------------------------
// NetCDF export
// -----------------------------------------------------------------------------
//
// Downloads a backend-generated NetCDF file.
//
// Returns:
// {
//   blob,
//   filename,
//   contentType
// }
// -----------------------------------------------------------------------------

export async function netcdfExport(
  params = {},
  options = {}
) {
  const response = await apiRequest(
    "/export/netcdf",
    {
      ...options,
      method: "GET",
      query: params,
      responseType: "blob",
    }
  );

  const headers = response?.headers;

  // apiRequest returns the Blob itself for responseType="blob".
  // Fetch headers are therefore not available here.
  //
  // Preserve compatibility by returning the blob directly when the
  // response does not expose headers.
  if (response instanceof Blob) {
    return {
      blob: response,
      filename: "neer-reconstruction.nc",
      contentType:
        response.type ||
        "application/x-netcdf",
    };
  }

  return response;
}

// -----------------------------------------------------------------------------
// Public exports
// -----------------------------------------------------------------------------

export default {
  health,
  modelInfo,
  dates,
  reconstruct,
  runReconstruction,
  reconstructGrid,
  profile,
  embedding,
  metrics,
  argoEvaluation,
  explainability,
  dataQuality,
  netcdfExport,
};

