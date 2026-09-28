// NEER Frontend — api.js test suite  (Phase 33D)
//
// Exercises the centralized request helper (`apiRequest`/`get`) once, then
// each endpoint function only for what's specific to it (its path and how it
// maps its arguments onto query parameters) — the shared mechanics (error
// normalization, encoding, timeouts, ...) are proven once against `get()`
// directly rather than re-proven per endpoint.
//
// `global.fetch` is replaced with a `vi.fn()` for every test — nothing here
// makes a real network call. `NEXT_PUBLIC_API_URL` is set in `beforeEach`
// and explicitly deleted in the "missing env var" tests.
// -----------------------------------------------------------------------------
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  API_ERROR_CODES,
  ApiError,
  apiRequest,
  argoEvaluation,
  categorizeApiError,
  dataQuality,
  dates,
  embedding,
  explainability,
  get,
  health,
  metrics,
  modelInfo,
  netcdfExport,
  profile,
  reconstruct,
  reconstructGrid,
} from "./api.js";

const BASE_URL = "http://localhost:8000";

/** Builds a fetch-Response-shaped object good enough for api.js's needs. */
function mockResponse({ ok = true, status = 200, statusText = "OK", bodyText = "", headers = {} } = {}) {
  const lowerHeaders = Object.fromEntries(Object.entries(headers).map(([k, v]) => [k.toLowerCase(), v]));
  return {
    ok,
    status,
    statusText,
    headers: { get: (name) => lowerHeaders[name.toLowerCase()] ?? null },
    text: () => Promise.resolve(bodyText),
    blob: () => Promise.resolve(new Blob([bodyText || ""])),
  };
}

beforeEach(() => {
  process.env.NEXT_PUBLIC_API_URL = BASE_URL;
  global.fetch = vi.fn();
});

afterEach(() => {
  delete process.env.NEXT_PUBLIC_API_URL;
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// Successful requests
// ---------------------------------------------------------------------------
describe("successful requests", () => {
  it("returns the parsed JSON body on 200", async () => {
    global.fetch.mockResolvedValue(
      mockResponse({ bodyText: JSON.stringify({ status: "ok", components: {} }) })
    );
    await expect(health()).resolves.toEqual({ status: "ok", components: {} });
  });

  it("returns null for an empty (e.g. 204) body", async () => {
    global.fetch.mockResolvedValue(mockResponse({ status: 204, statusText: "No Content", bodyText: "" }));
    await expect(get("/whatever")).resolves.toBeNull();
  });

  it("hits GET with no body and an Accept header", async () => {
    global.fetch.mockResolvedValue(mockResponse({ bodyText: "{}" }));
    await health();
    const [, init] = global.fetch.mock.calls[0];
    expect(init.method).toBe("GET");
    expect(init.body).toBeUndefined();
    expect(init.headers.Accept).toBe("application/json");
  });

  it("passes arrays / non-object JSON bodies straight through unchanged", async () => {
    global.fetch.mockResolvedValue(mockResponse({ bodyText: JSON.stringify(["a", "b"]) }));
    await expect(get("/list-endpoint")).resolves.toEqual(["a", "b"]);
  });
});

// ---------------------------------------------------------------------------
// Query parameter construction
// ---------------------------------------------------------------------------
describe("query parameter construction", () => {
  it("builds the base URL from NEXT_PUBLIC_API_URL with exactly one slash", async () => {
    global.fetch.mockResolvedValue(mockResponse({ bodyText: "{}" }));
    await health();
    expect(global.fetch.mock.calls[0][0]).toBe(`${BASE_URL}/health`);
  });

  it("encodes required numeric/date params for reconstruct()", async () => {
    global.fetch.mockResolvedValue(mockResponse({ bodyText: "{}" }));
    await reconstruct({ lat: 12.5, lon: -45.25, date: "2020-01-15", depth: 500 });
    const url = new URL(global.fetch.mock.calls[0][0]);
    expect(url.pathname).toBe("/reconstruct");
    expect(url.searchParams.get("lat")).toBe("12.5");
    expect(url.searchParams.get("lon")).toBe("-45.25");
    expect(url.searchParams.get("date")).toBe("2020-01-15");
    expect(url.searchParams.get("depth")).toBe("500");
  });

  it("formats a Date instance as YYYY-MM-DD in UTC", async () => {
    global.fetch.mockResolvedValue(mockResponse({ bodyText: "{}" }));
    await profile({ lat: 0, lon: 0, date: new Date("2021-06-01T00:00:00Z") });
    const url = new URL(global.fetch.mock.calls[0][0]);
    expect(url.searchParams.get("date")).toBe("2021-06-01");
  });

  it("omits an optional param that is undefined (reconstructGrid without depth)", async () => {
    global.fetch.mockResolvedValue(mockResponse({ bodyText: "{}" }));
    await reconstructGrid({ latMin: -10, latMax: 10, lonMin: 0, lonMax: 20, date: "2020-01-15" });
    const url = new URL(global.fetch.mock.calls[0][0]);
    expect(url.searchParams.has("depth")).toBe(false);
    expect(url.searchParams.get("lat_min")).toBe("-10");
    expect(url.searchParams.get("lat_max")).toBe("10");
  });

  it("includes an optional param that is provided (reconstructGrid with depth)", async () => {
    global.fetch.mockResolvedValue(mockResponse({ bodyText: "{}" }));
    await reconstructGrid({ latMin: -10, latMax: 10, lonMin: 0, lonMax: 20, date: "2020-01-15", depth: 250 });
    const url = new URL(global.fetch.mock.calls[0][0]);
    expect(url.searchParams.get("depth")).toBe("250");
  });

  it("percent-encodes characters that need it", async () => {
    global.fetch.mockResolvedValue(mockResponse({ bodyText: "{}" }));
    await get("/search", { q: "a b&c=d" });
    const url = new URL(global.fetch.mock.calls[0][0]);
    expect(url.searchParams.get("q")).toBe("a b&c=d"); // URL decodes it back for us
    expect(global.fetch.mock.calls[0][0]).toContain("a+b%26c%3Dd");
  });

  it("never serializes null/undefined as the literal string", async () => {
    global.fetch.mockResolvedValue(mockResponse({ bodyText: "{}" }));
    await get("/thing", { a: 1, b: undefined, c: null });
    const url = new URL(global.fetch.mock.calls[0][0]);
    expect(url.searchParams.has("b")).toBe(false);
    expect(url.searchParams.has("c")).toBe(false);
    expect(url.searchParams.get("a")).toBe("1");
  });
});

// ---------------------------------------------------------------------------
// HTTP errors
// ---------------------------------------------------------------------------
describe("HTTP errors", () => {
  it("normalizes a 400 (invalid_parameter) from the backend's {error, detail} shape", async () => {
    global.fetch.mockResolvedValue(
      mockResponse({
        ok: false,
        status: 400,
        statusText: "Bad Request",
        bodyText: JSON.stringify({ error: "invalid_parameter", detail: "lat_min >= lat_max" }),
      })
    );
    await expect(get("/reconstruct/grid")).rejects.toMatchObject({
      code: "invalid_parameter",
      status: 400,
      message: "lat_min >= lat_max",
    });
  });

  it("normalizes a 404 (date_not_found)", async () => {
    global.fetch.mockResolvedValue(
      mockResponse({
        ok: false,
        status: 404,
        bodyText: JSON.stringify({ error: "date_not_found", detail: "2099-01-01 is not available" }),
      })
    );
    await expect(reconstruct({ lat: 0, lon: 0, date: "2099-01-01", depth: 0 })).rejects.toMatchObject({
      code: "date_not_found",
      status: 404,
    });
  });

  it("normalizes FastAPI's default 422 validation shape ({ detail: [...] })", async () => {
    const detail = [{ loc: ["query", "lat"], msg: "field required", type: "value_error.missing" }];
    global.fetch.mockResolvedValue(
      mockResponse({ ok: false, status: 422, bodyText: JSON.stringify({ detail }) })
    );
    const error = await get("/reconstruct").catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.code).toBe(API_ERROR_CODES.VALIDATION);
    expect(error.status).toBe(422);
    expect(error.details).toEqual(detail);
  });

  it("normalizes a 500 (inference_failed)", async () => {
    global.fetch.mockResolvedValue(
      mockResponse({
        ok: false,
        status: 500,
        bodyText: JSON.stringify({ error: "inference_failed", detail: "forward pass raised" }),
      })
    );
    await expect(get("/reconstruct")).rejects.toMatchObject({ code: "inference_failed", status: 500 });
  });

  it("normalizes a 503 (model_unavailable) with extra fields preserved on .details", async () => {
    global.fetch.mockResolvedValue(
      mockResponse({
        ok: false,
        status: 503,
        bodyText: JSON.stringify({ error: "model_unavailable", detail: "no checkpoint loaded", retryable: true }),
      })
    );
    const error = await modelInfo().catch((e) => e);
    expect(error.code).toBe("model_unavailable");
    expect(error.details).toEqual({ retryable: true });
  });

  it("falls back to a generic HTTP error for an unrecognized error body", async () => {
    global.fetch.mockResolvedValue(mockResponse({ ok: false, status: 418, bodyText: "" }));
    await expect(get("/teapot")).rejects.toMatchObject({ code: API_ERROR_CODES.HTTP, status: 418 });
  });
});

// ---------------------------------------------------------------------------
// Invalid / unexpected response bodies
// ---------------------------------------------------------------------------
describe("malformed responses", () => {
  it("normalizes a 200 with a body that claims JSON but isn't parseable", async () => {
    global.fetch.mockResolvedValue(mockResponse({ ok: true, status: 200, bodyText: "{not valid json" }));
    await expect(get("/broken")).rejects.toMatchObject({ code: API_ERROR_CODES.INVALID_RESPONSE });
  });

  it("normalizes an error response whose body isn't parseable either", async () => {
    global.fetch.mockResolvedValue(mockResponse({ ok: false, status: 500, bodyText: "<html>oops</html>" }));
    await expect(get("/broken")).rejects.toMatchObject({ code: API_ERROR_CODES.INVALID_RESPONSE, status: 500 });
  });
});

// ---------------------------------------------------------------------------
// Network / timeout / abort failures
// ---------------------------------------------------------------------------
describe("network and timeout failures", () => {
  it("normalizes a rejected fetch() (DNS/connection failure) as a network error", async () => {
    global.fetch.mockRejectedValue(new TypeError("Failed to fetch"));
    const error = await health().catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.code).toBe(API_ERROR_CODES.NETWORK);
    expect(error.status).toBeNull();
  });

  it("normalizes an internally-triggered abort as a timeout", async () => {
    global.fetch.mockImplementation(
      (_url, init) =>
        new Promise((_resolve, reject) => {
          init.signal.addEventListener("abort", () => {
            const err = new Error("aborted");
            err.name = "AbortError";
            reject(err);
          });
        })
    );
    await expect(apiRequest("/slow", { timeoutMs: 5 })).rejects.toMatchObject({
      code: API_ERROR_CODES.TIMEOUT,
    });
  });

  it("normalizes a caller-supplied AbortSignal firing as 'aborted', not 'timeout'", async () => {
    const controller = new AbortController();
    global.fetch.mockImplementation(
      (_url, init) =>
        new Promise((_resolve, reject) => {
          init.signal.addEventListener("abort", () => {
            const err = new Error("aborted");
            err.name = "AbortError";
            reject(err);
          });
        })
    );
    const pending = apiRequest("/slow", { signal: controller.signal });
    controller.abort();
    await expect(pending).rejects.toMatchObject({ code: API_ERROR_CODES.ABORTED });
  });
});

// ---------------------------------------------------------------------------
// Missing NEXT_PUBLIC_API_URL
// ---------------------------------------------------------------------------
describe("missing NEXT_PUBLIC_API_URL", () => {
  it("throws a config_error and never calls fetch (no hard-coded fallback)", async () => {
    delete process.env.NEXT_PUBLIC_API_URL;
    const error = await health().catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.code).toBe(API_ERROR_CODES.CONFIG);
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("also rejects a blank/whitespace-only value", async () => {
    process.env.NEXT_PUBLIC_API_URL = "   ";
    const error = await health().catch((e) => e);
    expect(error.code).toBe(API_ERROR_CODES.CONFIG);
    expect(global.fetch).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// Per-endpoint behavior (Phase 33C functions)
// ---------------------------------------------------------------------------
describe("endpoint function behavior", () => {
  beforeEach(() => {
    global.fetch.mockResolvedValue(mockResponse({ bodyText: "{}" }));
  });

  it("embedding() requires date and hits GET /embedding", async () => {
    await embedding({ date: "2020-01-15" });
    const url = new URL(global.fetch.mock.calls[0][0]);
    expect(url.pathname).toBe("/embedding");
    expect(url.searchParams.get("date")).toBe("2020-01-15");
  });

  it("metrics() omits split when not provided (backend applies its own default)", async () => {
    await metrics();
    const url = new URL(global.fetch.mock.calls[0][0]);
    expect(url.pathname).toBe("/metrics");
    expect(url.searchParams.has("split")).toBe(false);
  });

  it("metrics() sends split when provided", async () => {
    await metrics({ split: "val" });
    const url = new URL(global.fetch.mock.calls[0][0]);
    expect(url.searchParams.get("split")).toBe("val");
  });

  it("argoEvaluation() defaults to no demo param (real validation, not the demo opt-in)", async () => {
    await argoEvaluation();
    const url = new URL(global.fetch.mock.calls[0][0]);
    expect(url.pathname).toBe("/evaluation/argo");
    expect(url.searchParams.has("demo")).toBe(false);
  });

  it("argoEvaluation({ demo: true }) explicitly opts into the demo pipeline check", async () => {
    await argoEvaluation({ demo: true });
    const url = new URL(global.fetch.mock.calls[0][0]);
    expect(url.searchParams.get("demo")).toBe("true");
  });

  it("explainability() sends date and omits depth when not given", async () => {
    await explainability({ date: "2020-01-15" });
    const url = new URL(global.fetch.mock.calls[0][0]);
    expect(url.pathname).toBe("/explainability");
    expect(url.searchParams.get("date")).toBe("2020-01-15");
    expect(url.searchParams.has("depth")).toBe(false);
  });

  it("explainability() includes depth when given", async () => {
    await explainability({ date: "2020-01-15", depth: 100 });
    const url = new URL(global.fetch.mock.calls[0][0]);
    expect(url.searchParams.get("depth")).toBe("100");
  });

  it("dataQuality() takes no parameters", async () => {
    await dataQuality();
    const [calledUrl] = global.fetch.mock.calls[0];
    expect(calledUrl).toBe(`${BASE_URL}/data/quality`);
  });

  it("dates() and modelInfo() hit their fixed, parameter-less paths", async () => {
    // Phase 36A: both now validate their body, so the mocks return valid ones.
    global.fetch
      .mockResolvedValueOnce(mockResponse({ bodyText: JSON.stringify({ dates: ["2020-01-01"] }) }))
      .mockResolvedValueOnce(mockResponse({ bodyText: JSON.stringify({ architecture: {} }) }));
    await dates();
    await modelInfo();
    expect(global.fetch.mock.calls[0][0]).toBe(`${BASE_URL}/dates`);
    expect(global.fetch.mock.calls[1][0]).toBe(`${BASE_URL}/model/info`);
  });
});

// ---------------------------------------------------------------------------
// netcdfExport() — the one binary/blob endpoint
// ---------------------------------------------------------------------------
describe("netcdfExport()", () => {
  it("returns a Blob and reads the filename from Content-Disposition", async () => {
    global.fetch.mockResolvedValue(
      mockResponse({
        bodyText: "fake-netcdf-bytes",
        headers: {
          "content-disposition": 'attachment; filename="neer_reconstruct_2020-01-15_500m.nc"',
          "content-type": "application/x-netcdf",
        },
      })
    );
    const result = await netcdfExport({
      latMin: -10,
      latMax: 10,
      lonMin: 0,
      lonMax: 20,
      date: "2020-01-15",
      depth: 500,
    });
    expect(result.blob).toBeInstanceOf(Blob);
    expect(result.filename).toBe("neer_reconstruct_2020-01-15_500m.nc");
    expect(result.contentType).toBe("application/x-netcdf");

    const [calledUrl] = global.fetch.mock.calls[0];
    const url = new URL(calledUrl);
    expect(url.pathname).toBe("/reconstruct/netcdf");
    expect(url.searchParams.get("depth")).toBe("500");
  });

  it("falls back to a locally-built filename when Content-Disposition is missing", async () => {
    global.fetch.mockResolvedValue(mockResponse({ bodyText: "fake-netcdf-bytes" }));
    const result = await netcdfExport({
      latMin: -10,
      latMax: 10,
      lonMin: 0,
      lonMax: 20,
      date: "2020-01-15",
    });
    expect(result.filename).toBe("neer_reconstruct_2020-01-15.nc");
  });

  it("still normalizes a non-ok response as a regular ApiError (not a blob)", async () => {
    global.fetch.mockResolvedValue(
      mockResponse({
        ok: false,
        status: 503,
        bodyText: JSON.stringify({ error: "netcdf_unavailable", detail: "xarray not installed" }),
      })
    );
    await expect(
      netcdfExport({ latMin: -10, latMax: 10, lonMin: 0, lonMax: 20, date: "2020-01-15" })
    ).rejects.toMatchObject({ code: "netcdf_unavailable", status: 503 });
  });
});

// ---------------------------------------------------------------------------
// Phase 36A — dates() / modelInfo() response validation, categorizeApiError
// ---------------------------------------------------------------------------
describe("dates() response validation (Phase 36A)", () => {
  const ok = (body) => global.fetch.mockResolvedValue(mockResponse({ bodyText: JSON.stringify(body) }));

  it("resolves with the backend body unchanged on a valid response", async () => {
    const body = { dates: ["2020-01-01", "2020-01-02"], count: 2, min_date: "2020-01-01", max_date: "2020-01-02", data_mode: "REAL", is_synthetic: false };
    ok(body);
    await expect(dates()).resolves.toEqual(body);
  });

  it("resolves (does not throw) for an empty date list — that's the caller's empty state", async () => {
    ok({ dates: [], count: 0, min_date: null, max_date: null, data_mode: "REAL", is_synthetic: false });
    await expect(dates()).resolves.toMatchObject({ dates: [] });
  });

  it.each([
    ["an empty body", ""],
    ["a body without `dates`", JSON.stringify({ count: 0 })],
    ["a non-array `dates`", JSON.stringify({ dates: "2020-01-01" })],
    ["a malformed date entry", JSON.stringify({ dates: ["2020-01-01", "nope"] })],
  ])("throws invalid_response for %s", async (_l, bodyText) => {
    global.fetch.mockResolvedValue(mockResponse({ bodyText }));
    const error = await dates().catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.code).toBe(API_ERROR_CODES.INVALID_RESPONSE);
    expect(categorizeApiError(error)).toBe("invalid_response");
  });

  it("passes the backend's data_unavailable 503 through as a normalized api error", async () => {
    global.fetch.mockResolvedValue(
      mockResponse({ ok: false, status: 503, statusText: "Service Unavailable", bodyText: JSON.stringify({ error: "data_unavailable", detail: "no dataset is loaded" }) })
    );
    const error = await dates().catch((e) => e);
    expect(error).toMatchObject({ code: "data_unavailable", status: 503 });
    expect(categorizeApiError(error)).toBe("api");
  });

  it("surfaces an unreachable backend as a network error", async () => {
    global.fetch.mockRejectedValue(new TypeError("Failed to fetch"));
    const error = await dates().catch((e) => e);
    expect(error.code).toBe(API_ERROR_CODES.NETWORK);
    expect(categorizeApiError(error)).toBe("network");
  });
});

describe("modelInfo() response validation (Phase 36A)", () => {
  it("returns a valid object body unchanged", async () => {
    const body = { architecture: { depths: [0, 5], num_depths: 2 }, runtime: {}, checkpoint: {} };
    global.fetch.mockResolvedValue(mockResponse({ bodyText: JSON.stringify(body) }));
    await expect(modelInfo()).resolves.toEqual(body);
  });
  it.each([
    ["empty body", ""],
    ["array body", "[]"],
    ["string body", JSON.stringify("ok")],
  ])("throws invalid_response for %s", async (_l, bodyText) => {
    global.fetch.mockResolvedValue(mockResponse({ bodyText }));
    const error = await modelInfo().catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.code).toBe(API_ERROR_CODES.INVALID_RESPONSE);
  });
});

describe("categorizeApiError", () => {
  it("maps codes to UI categories", () => {
    const mk = (code, status = null) => new ApiError({ code, message: "m", status });
    expect(categorizeApiError(mk(API_ERROR_CODES.TIMEOUT))).toBe("timeout");
    expect(categorizeApiError(mk(API_ERROR_CODES.CONFIG))).toBe("config");
    expect(categorizeApiError(mk(API_ERROR_CODES.ABORTED))).toBe("aborted");
    expect(categorizeApiError(mk("model_unavailable", 503))).toBe("api");
    expect(categorizeApiError(mk("weird", null))).toBe("unknown");
    expect(categorizeApiError(new Error("x"))).toBe("unknown");
    expect(categorizeApiError(undefined)).toBe("unknown");
  });
});