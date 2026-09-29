import { describe, expect, it } from "vitest";
import { formatAnomaly, summarizeFiniteValues, validateGridAnomalyResponse } from "./anomaly.js";
import { validatePointReconstruction } from "./reconstructionResponse.js";

const expectedPoint = { lat: 15.25, lon: 88.5, date: "2024-05-01", depth: 200 };
const point = (overrides = {}) => ({ mode: "point", ...expectedPoint, temperature: 10, climatology: 8, anomaly: 2,
  embedding_dim: 8, data_mode: "REAL", latency_ms: 0, cache_hit: false, notes: [], ...overrides });

describe("canonical anomaly validation", () => {
  it.each([[10, 8, 2], [6, 8, -2], [8, 8, 0]])("accepts model output %s - climatology %s = anomaly %s", (temperature, climatology, anomaly) => {
    expect(validatePointReconstruction(point({ temperature, climatology, anomaly }), { date: expectedPoint.date }).ok).toBe(true);
  });
  it("rejects mismatched arithmetic and only accepts unavailable triplets together", () => {
    expect(validatePointReconstruction(point({ anomaly: 1.5 }), { date: expectedPoint.date }).ok).toBe(false);
    expect(validatePointReconstruction(point({ temperature: null, climatology: null, anomaly: null }), { date: expectedPoint.date }).ok).toBe(true);
    expect(validatePointReconstruction(point({ temperature: 10, climatology: null, anomaly: 2 }), { date: expectedPoint.date }).ok).toBe(false);
  });
});

describe("anomaly summaries and formatting", () => {
  it("summarizes only actual finite values in one nested dataset", () => {
    expect(summarizeFiniteValues([-2, null, 0, NaN, 2, Infinity])).toEqual({ min: -2, max: 2, mean: 0, validCells: 3 });
  });
  it("returns N/A statistics for empty data and preserves zero anomaly formatting", () => {
    expect(summarizeFiniteValues([null, NaN])).toEqual({ min: null, max: null, mean: null, validCells: 0 });
    expect(formatAnomaly(0)).toBe("0.00 °C");
    expect(formatAnomaly(0.37482917)).toBe("+0.37 °C");
    expect(formatAnomaly(-0.42)).toBe("-0.42 °C");
    expect(formatAnomaly(Infinity)).toBe("N/A");
  });
});

describe("aligned map-grid anomaly response", () => {
  const expected = { date: "2024-05-01", depth: 200, latMin: 10, latMax: 10.25, lonMin: 60, lonMax: 60.25 };
  const grid = { date: expected.date, depth: 200, lat: [10, 10.25], lon: [60, 60.25],
    temperature: [[10, 12], [null, 8]], climatology: [[8, 10], [null, 8]], anomaly: [[2, 2], [null, 0]] };
  it("accepts matching coordinates and backend fields including zero", () => {
    expect(validateGridAnomalyResponse(grid, expected)).toMatchObject({ ok: true });
  });
  it("rejects stale or arithmetically inconsistent fields", () => {
    expect(validateGridAnomalyResponse({ ...grid, date: "2024-05-02" }, expected).ok).toBe(false);
    expect(validateGridAnomalyResponse({ ...grid, anomaly: [[1, 2], [null, 0]] }, expected).ok).toBe(false);
  });
});
