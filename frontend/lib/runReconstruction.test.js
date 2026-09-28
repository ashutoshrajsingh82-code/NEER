// NEER Frontend — runReconstruction test suite  (Phase 37A)
//
// `global.fetch` is mocked for every test — nothing here touches a network.
// Response fixtures mirror the backend's `PointReconstructionResponse`
// (backend/app/schemas.py); error fixtures mirror `NeerApiError.body()`
// (backend/app/errors.py).
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { API_ERROR_CODES, ApiError, categorizeApiError, runReconstruction } from "./api.js";
import { validatePointReconstruction } from "./reconstructionResponse.js";

const BASE_URL = "http://localhost:8000";
const INPUT = { date: "2020-01-15", depth: 100, latitude: 12.5, longitude: 72.25 };

const validBody = (overrides = {}) => ({
  mode: "point",
  lat: 12.5,
  lon: 72.25,
  date: "2020-01-15",
  depth: 100,
  temperature: 24.31,
  anomaly: -0.42,
  climatology: 24.73,
  embedding_dim: 128,
  data_mode: "reconstructed",
  latency_ms: 12.5,
  cache_hit: false,
  notes: [],
  ...overrides,
});

function mockResponse({ ok = true, status = 200, statusText = "OK", bodyText = "" } = {}) {
  return {
    ok,
    status,
    statusText,
    headers: { get: () => null },
    text: () => Promise.resolve(bodyText),
    blob: () => Promise.resolve(new Blob([bodyText])),
  };
}
const okJson = (body) => mockResponse({ bodyText: JSON.stringify(body) });
const errJson = (status, error, detail) =>
  mockResponse({ ok: false, status, statusText: "err", bodyText: JSON.stringify({ error, detail }) });

beforeEach(() => {
  process.env.NEXT_PUBLIC_API_URL = BASE_URL;
  global.fetch = vi.fn();
});
afterEach(() => {
  delete process.env.NEXT_PUBLIC_API_URL;
  vi.restoreAllMocks();
});

describe("runReconstruction — request", () => {
  it("calls GET /reconstruct with date, depth, lat and lon, and no body", async () => {
    global.fetch.mockResolvedValue(okJson(validBody()));
    await runReconstruction(INPUT);
    expect(global.fetch).toHaveBeenCalledTimes(1);
    const [rawUrl, init] = global.fetch.mock.calls[0];
    const url = new URL(rawUrl);
    expect(url.origin).toBe(BASE_URL);
    expect(url.pathname).toBe("/reconstruct");
    expect(Object.fromEntries(url.searchParams)).toEqual({
      lat: "12.5",
      lon: "72.25",
      date: "2020-01-15",
      depth: "100",
    });
    expect(init.method).toBe("GET");
    expect(init.body).toBeUndefined();
  });

  it("sends depth 0 and a Date input (formatted in UTC) correctly", async () => {
    global.fetch.mockResolvedValue(okJson(validBody({ depth: 0 })));
    await runReconstruction({ ...INPUT, depth: 0, date: new Date(Date.UTC(2020, 0, 15)) });
    const url = new URL(global.fetch.mock.calls[0][0]);
    expect(url.searchParams.get("depth")).toBe("0");
    expect(url.searchParams.get("date")).toBe("2020-01-15");
  });

  it.each([
    ["missing date", { date: undefined }],
    ["malformed date", { date: "15/01/2020" }],
    ["impossible date", { date: "2020-02-30" }],
    ["missing depth", { depth: undefined }],
    ["negative depth", { depth: -5 }],
    ["NaN depth", { depth: NaN }],
    ["missing latitude", { latitude: undefined }],
    ["latitude out of range", { latitude: 95 }],
    ["missing longitude", { longitude: undefined }],
    ["longitude out of range", { longitude: 400 }],
    ["string coordinates", { latitude: "12.5" }],
  ])("rejects %s without sending any request", async (_label, patch) => {
    const error = await runReconstruction({ ...INPUT, ...patch }).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.code).toBe(API_ERROR_CODES.VALIDATION);
    expect(error.status).toBeNull();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("rejects being called with no arguments", async () => {
    await expect(runReconstruction()).rejects.toMatchObject({ code: API_ERROR_CODES.VALIDATION });
    expect(global.fetch).not.toHaveBeenCalled();
  });
});

describe("runReconstruction — success", () => {
  it("returns the validated backend body unchanged", async () => {
    const body = validBody({ notes: ["depth snapped from 90 to 100"] });
    global.fetch.mockResolvedValue(okJson(body));
    await expect(runReconstruction(INPUT)).resolves.toEqual(body);
  });

  it("accepts a null climatology (nullable in the backend schema)", async () => {
    global.fetch.mockResolvedValue(okJson(validBody({ climatology: null })));
    const result = await runReconstruction(INPUT);
    expect(result.climatology).toBeNull();
    expect(result.temperature).toBe(24.31);
  });

  it("returns the snapped depth the backend used, not the requested one", async () => {
    global.fetch.mockResolvedValue(okJson(validBody({ depth: 100 })));
    const result = await runReconstruction({ ...INPUT, depth: 90 });
    expect(result.depth).toBe(100);
  });
});

describe("runReconstruction — HTTP errors", () => {
  it("HTTP 400 -> invalid_parameter", async () => {
    global.fetch.mockResolvedValue(errJson(400, "invalid_parameter", "lat/lon outside the NEER domain"));
    const error = await runReconstruction(INPUT).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ code: "invalid_parameter", status: 400 });
    expect(error.message).toBe("lat/lon outside the NEER domain");
    expect(categorizeApiError(error)).toBe("api");
  });

  it("HTTP 404 -> date_not_found", async () => {
    global.fetch.mockResolvedValue(errJson(404, "date_not_found", "2020-01-15 is not available"));
    await expect(runReconstruction(INPUT)).rejects.toMatchObject({
      code: "date_not_found",
      status: 404,
      message: "2020-01-15 is not available",
    });
  });

  it("HTTP 500 -> inference_failed", async () => {
    global.fetch.mockResolvedValue(errJson(500, "inference_failed", "reconstruction failed: boom"));
    await expect(runReconstruction(INPUT)).rejects.toMatchObject({ code: "inference_failed", status: 500 });
  });

  it("HTTP 500 with a non-JSON body -> invalid_response, still an error", async () => {
    global.fetch.mockResolvedValue(
      mockResponse({ ok: false, status: 500, statusText: "Internal Server Error", bodyText: "<html>oops</html>" })
    );
    await expect(runReconstruction(INPUT)).rejects.toMatchObject({
      code: API_ERROR_CODES.INVALID_RESPONSE,
      status: 500,
    });
  });
});

describe("runReconstruction — network failure", () => {
  it("fetch rejecting -> network_error (status null)", async () => {
    global.fetch.mockRejectedValue(new TypeError("Failed to fetch"));
    const error = await runReconstruction(INPUT).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ code: API_ERROR_CODES.NETWORK, status: null });
    expect(categorizeApiError(error)).toBe("network");
  });

  it("missing NEXT_PUBLIC_API_URL -> config_error", async () => {
    delete process.env.NEXT_PUBLIC_API_URL;
    await expect(runReconstruction(INPUT)).rejects.toMatchObject({ code: API_ERROR_CODES.CONFIG });
    expect(global.fetch).not.toHaveBeenCalled();
  });
});

describe("runReconstruction — malformed / incomplete responses are errors", () => {
  const expectInvalid = async (body) => {
    global.fetch.mockResolvedValue(typeof body === "string" ? mockResponse({ bodyText: body }) : okJson(body));
    const error = await runReconstruction(INPUT).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.code).toBe(API_ERROR_CODES.INVALID_RESPONSE);
    return error;
  };

  it("unparseable JSON", async () => {
    await expectInvalid("{not json");
  });

  it("empty 200 body", async () => {
    await expectInvalid("");
  });

  it("JSON that isn't an object", async () => {
    await expectInvalid([1, 2, 3]);
    await expectInvalid("42");
  });

  it.each(["temperature", "anomaly", "date", "depth", "lat", "lon", "data_mode", "cache_hit", "notes", "latency_ms"])(
    "missing %s",
    async (field) => {
      const body = validBody();
      delete body[field];
      const error = await expectInvalid(body);
      expect(error.message).toContain(field);
      expect(error.details.fields).toContain(field);
    }
  );

  it("missing climatology key", async () => {
    const body = validBody();
    delete body.climatology;
    await expectInvalid(body);
  });

  it("null temperature is not treated as 'no data'", async () => {
    await expectInvalid(validBody({ temperature: null }));
  });

  it.each([
    ["string temperature", { temperature: "24.3" }],
    ["NaN-as-null anomaly", { anomaly: null }],
    ["non-boolean cache_hit", { cache_hit: "false" }],
    ["notes not an array", { notes: "none" }],
    ["malformed date", { date: "2020/01/15" }],
    ["negative depth", { depth: -1 }],
  ])("invalid field: %s", async (_label, patch) => {
    await expectInvalid(validBody(patch));
  });

  it("response for a different date than requested", async () => {
    const error = await expectInvalid(validBody({ date: "2020-01-16" }));
    expect(error.message).toContain("2020-01-16");
  });

  it("never returns a partial/defaulted object on failure", async () => {
    global.fetch.mockResolvedValue(okJson({ temperature: 20 }));
    await expect(runReconstruction(INPUT)).rejects.toBeInstanceOf(ApiError);
  });
});

describe("validatePointReconstruction", () => {
  it("passes a complete body through by reference", () => {
    const body = validBody();
    const result = validatePointReconstruction(body, { date: "2020-01-15" });
    expect(result.ok).toBe(true);
    expect(result.value).toBe(body);
  });

  it("does not check the date echo when none is expected", () => {
    expect(validatePointReconstruction(validBody({ date: "1999-12-31" })).ok).toBe(true);
  });

  it("rejects null / arrays / primitives", () => {
    for (const bad of [null, undefined, [], "x", 3]) {
      expect(validatePointReconstruction(bad).ok).toBe(false);
    }
  });

  it("rejects non-finite numbers", () => {
    expect(validatePointReconstruction(validBody({ temperature: Infinity })).ok).toBe(false);
    expect(validatePointReconstruction(validBody({ anomaly: NaN })).ok).toBe(false);
  });
});