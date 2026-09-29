import { describe, expect, it } from "vitest";
import { validateProfileResponse, valueOrNA, profileSeries, profilePointAtDepth, profileCsv } from "./profileResponse.js";

const expected = { lat: 15.25, lon: 88.5, date: "2024-05-01" };
const profile = (overrides = {}) => ({
  mode: "profile", lat: expected.lat, lon: expected.lon, date: expected.date,
  depths: [0, 100, 500], temperature: [28.4, 24.1, null], anomaly: [1.2, -0.8, null],
  climatology: [27.2, 24.9, null], embedding_dim: 32, data_mode: "reconstructed",
  latency_ms: 20, cache_hit: false, notes: [], ...overrides,
});

describe("profile response validation", () => {
  it("accepts an aligned response for the requested point and date", () => {
    expect(validateProfileResponse(profile(), expected)).toMatchObject({ ok: true });
  });
  it.each([
    ["different date", {}, { date: "2024-04-30" }],
    ["different latitude", {}, { lat: 15.5 }],
    ["different longitude", {}, { lon: 89 }],
    ["misaligned temperature axis", { temperature: [28.4] }, {}],
    ["non-finite depth", { depths: [0, Infinity, 500] }, {}],
    ["malformed climatology", { climatology: [1] }, {}],
  ])("rejects %s", (_label, dataPatch, expectedPatch) => {
    expect(validateProfileResponse(profile(dataPatch), { ...expected, ...expectedPatch }).ok).toBe(false);
  });
  it("accepts missing paired values as unavailable when climatology is missing", () => {
    expect(validateProfileResponse(profile({ climatology: null, temperature: [null, null, null], anomaly: [null, null, null] }), expected).ok).toBe(true);
  });
});

describe("N/A formatting", () => {
  it.each([null, undefined, NaN, Infinity, "", {}])("shows N/A for %s", (value) => {
    expect(valueOrNA(value)).toBe("N/A");
  });
  it("preserves valid zero values", () => {
    expect(valueOrNA(0)).toBe(0);
  });
});

describe("profile series and export helpers", () => {
  it("marks only series with actual numeric data as available", () => {
    expect(profileSeries(profile()).map(({ key, available }) => [key, available])).toEqual([
      ["neer", true], ["climatology", true], ["anomaly", true], ["argo", false],
    ]);
    expect(profileSeries(profile({ climatology: [null, null, null] }))[1].available).toBe(false);
  });
  it("returns exact-depth values without interpolating and marks missing series N/A", () => {
    expect(profilePointAtDepth(profile(), 100)).toEqual({ depth: 100, neer: 24.1, climatology: 24.9, anomaly: -0.8, argo: "N/A" });
    expect(profilePointAtDepth(profile(), 50)).toBeNull();
    expect(profilePointAtDepth(profile(), 500)).toMatchObject({ neer: "N/A", climatology: "N/A" });
  });
  it("exports real aligned values, metadata, and blank missing fields", () => {
    const csv = profileCsv(profile());
    expect(csv).toContain("# latitude,15.25");
    expect(csv).toContain("# date,2024-05-01");
    expect(csv).toContain("0,28.4,27.2,1.2,");
    expect(csv).toContain("500,,,,");
    expect(csv).not.toContain("500,0,0,0");
  });
});
