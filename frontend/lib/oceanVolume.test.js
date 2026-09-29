import { describe, expect, it } from "vitest";
import { validateOceanVolume, volumeColor, volumeValue } from "./oceanVolume.js";

const expected = { date: "2024-05-01", depths: [0, 20], latMin: 10, latMax: 10.25, lonMin: 60, lonMax: 60.25 };
const raw = (patch = {}) => ({ date: expected.date, depth: null, depths: [0, 20], lat: [10, 10.25], lon: [60, 60.25],
  temperature: [[[28, 22], [null, 21]], [[27, 20], [26, 19]]], anomaly: [[[1.2, -0.5], [null, -0.5]], [[1.2, -0.5], [1, -0.5]]],
  climatology: [[[26.8, 22.5], [null, 21.5]], [[25.8, 20.5], [25, 19.5]]], data_mode: "REAL", ...patch });

describe("3D ocean volume validation and values", () => {
  it("accepts real aligned grids and keeps coordinates and matrix dimensions", () => {
    expect(validateOceanVolume(raw(), expected).ok).toBe(true);
  });
  it("rejects a stale date, mismatched grid shape, or malformed anomaly", () => {
    expect(validateOceanVolume(raw({ date: "2024-05-02" }), expected).ok).toBe(false);
    expect(validateOceanVolume(raw({ temperature: [[[1]]] }), expected).ok).toBe(false);
    expect(validateOceanVolume(raw({ anomaly: [1] }), expected).ok).toBe(false);
  });
  it("rejects coordinates and depths outside the authoritative domain/configuration", () => {
    expect(validateOceanVolume(raw({ lat: [4, 10.25] }), expected).ok).toBe(false);
    expect(validateOceanVolume(raw({ depths: [0, 21] }), expected).ok).toBe(false);
  });
  it("indexes the backend's aligned latitude-longitude-depth fields", () => {
    const data = raw();
    expect(volumeValue(data, 1, 0, 1, "temperature")).toBe(20);
    expect(volumeValue(data, 0, 1, 0, "temperature")).toBeNull();
    expect(volumeValue(data, 1, 1, 0, "anomaly")).toBe(1);
  });
  it("leaves missing values without scientific colors and preserves valid zero", () => {
    expect(volumeColor(null, 0, 30)).toBeNull();
    expect(volumeColor(0, 0, 30)).toEqual([22, 78, 150]);
    expect(volumeColor(0, -1, 1, "anomaly")).toEqual([230, 237, 245]);
  });
});
