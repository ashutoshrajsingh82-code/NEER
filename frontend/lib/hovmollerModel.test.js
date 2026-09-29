import { describe, expect, it } from "vitest";
import { hovmollerColor, hovmollerCsv, normalizeHovmoller, validateHovmollerSelection } from "./hovmollerModel.js";

const dates = ["2024-05-01", "2024-05-03"];
const depths = [0, 5, 20, 100];
const request = { lat: 15.25, lon: 88.5, startDate: dates[0], endDate: dates[1], minDepth: 5, maxDepth: 100 };
const profiles = [
  { lat: 15.25, lon: 88.5, date: dates[0], depths, temperature: [28, 27, null, 12], climatology: [27, 28, null, 13], anomaly: [1, -1, null, -1], data_mode: "REAL" },
  { lat: 15.25, lon: 88.5, date: dates[1], depths, temperature: [29, 26, 18, 11], climatology: [28, 27, 20, 12], anomaly: [1, -1, -2, -1], data_mode: "REAL" },
];

describe("Hovmoller selection validation", () => {
  const check = (patch = {}) => validateHovmollerSelection({ ...request, availableDates: dates, availableDepths: depths, ...patch });
  it("accepts supported coordinates, dates, and depth bounds", () => expect(check().ok).toBe(true));
  it.each([[{ lat: 4.99 }, "latitude"], [{ lon: 105.01 }, "longitude"], [{ startDate: "2024-05-02" }, "startDate"], [{ startDate: dates[1], endDate: dates[0] }, "dateRange"], [{ minDepth: 6 }, "depthRange"], [{ minDepth: 100, maxDepth: 5 }, "depthRange"]])("rejects invalid selection %o", (patch, field) => expect(check(patch).field).toBe(field));
});

describe("Hovmoller time-depth matrix", () => {
  it("maps dates to rows, depths to columns, and retains null cells", () => {
    const result = normalizeHovmoller(profiles, dates, request);
    expect(result.ok).toBe(true);
    expect(result.value.dates).toEqual(dates);
    expect(result.value.depths).toEqual([5, 20, 100]);
    expect(result.value.values).toEqual([[27, null, 12], [26, 18, 11]]);
    expect(result.value.anomalyValues).toEqual([[-1, null, -1], [-1, -2, -1]]);
    expect(result.value.dataMode).toBe("REAL");
  });
  it("rejects responses that do not match the requested point and time", () => {
    expect(normalizeHovmoller([{ ...profiles[0], lat: 16 }, profiles[1]], dates, request).ok).toBe(false);
    expect(normalizeHovmoller(profiles, ["2024-05-02", dates[1]], request).ok).toBe(false);
  });
  it("rejects transposed or malformed axes instead of guessing", () => {
    expect(normalizeHovmoller([{ ...profiles[0], temperature: [1] }, profiles[1]], dates, request).ok).toBe(false);
    expect(normalizeHovmoller([{ ...profiles[0], depths: [0, 20, 5, 100] }, profiles[1]], dates, request).ok).toBe(false);
  });
  it("assigns colors only to finite temperatures", () => {
    expect(hovmollerColor(null, 0, 30)).toBeNull();
    expect(hovmollerColor(0, 0, 30)).toMatch(/^rgb\(/);
    expect(hovmollerColor(30, 0, 30)).not.toBe(hovmollerColor(0, 0, 30));
  });
  it("centers anomaly colors on neutral zero and preserves both signs", () => {
    expect(hovmollerColor(0, -2, 1, "anomaly")).toBe("rgba(247, 247, 247, 1)");
    expect(hovmollerColor(-1, -2, 1, "anomaly")).not.toBe(hovmollerColor(1, -2, 1, "anomaly"));
  });
  it("exports full context and blank cells without filling gaps", () => {
    const matrix = normalizeHovmoller(profiles, dates, request).value;
    const csv = hovmollerCsv(matrix);
    expect(csv).toContain("# latitude,15.25");
    expect(csv).toContain("# start_date,2024-05-01");
    expect(csv).toContain("2024-05-01,20,");
    expect(csv).toContain("2024-05-03,5,26");
    expect(csv).not.toContain("2024-05-01,20,0");
  });
});
