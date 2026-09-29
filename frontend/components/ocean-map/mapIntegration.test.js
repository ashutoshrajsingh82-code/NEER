import { describe, expect, it } from "vitest";
import { gridEdgeLines } from "./GridOverlay";
import {
  GRID_CACHE_LIMIT,
  MAX_GRID_POINTS,
  boundsContain,
  estimateCells,
  fieldContextMatches,
  gridCacheKey,
  isRequestableDepth,
  planRequestBounds,
  rememberInCache,
} from "./useOceanMapLayer";
import { OCEAN_DOMAIN, toGridSelection } from "@/lib/oceanDomain";

describe("gridEdgeLines — grid aligned with the scientific cells", () => {
  const { latMin, latMax, resolution } = OCEAN_DOMAIN;

  it("draws lines on cell edges (half a step from cell centers), never through centers", () => {
    const lines = gridEdgeLines(latMin, latMax, 10, 11, resolution, resolution);
    // cell centers are 10.00, 10.25, ...; edges are 9.875, 10.125, ...
    expect(lines).toContain(10.125);
    expect(lines).toContain(10.875);
    expect(lines).not.toContain(10);
    expect(lines).not.toContain(10.25);
  });

  it("every line is exactly midway between two adjacent cell centers", () => {
    for (const edge of gridEdgeLines(latMin, latMax, 5, 12, resolution, resolution)) {
      const offset = (edge - latMin) / resolution; // in cells, from the first center
      expect(Math.abs(offset - Math.round(offset))).toBeCloseTo(0.5, 6);
    }
  });

  it("never returns edges outside the domain", () => {
    const lines = gridEdgeLines(latMin, latMax, -10, 100, resolution, resolution);
    expect(Math.min(...lines)).toBeGreaterThanOrEqual(latMin);
    expect(Math.max(...lines)).toBeLessThanOrEqual(latMax);
  });

  it("coarser strides keep the same edge alignment", () => {
    const lines = gridEdgeLines(latMin, latMax, 5, 8, resolution * 4, resolution);
    expect(lines[0]).toBeCloseTo(5.875, 6); // 4.875 (first cell edge) + 1° blocks: first in-domain edge is 5.875
    for (let i = 1; i < lines.length; i += 1) expect(lines[i] - lines[i - 1]).toBeCloseTo(1, 6);
  });

  it("returns [] for invalid input", () => {
    expect(gridEdgeLines(5, 30, NaN, 10, 0.25, 0.25)).toEqual([]);
    expect(gridEdgeLines(5, 30, 5, 10, 0, 0.25)).toEqual([]);
  });
});

describe("request planning — fewer API calls", () => {
  it("snaps the viewport outward to whole degrees so nearby pans share a request", () => {
    const a = planRequestBounds({ latMin: 10.13, latMax: 14.2, lonMin: 60.4, lonMax: 64.9 });
    const b = planRequestBounds({ latMin: 10.5, latMax: 14.9, lonMin: 60.9, lonMax: 64.1 });
    expect(a).toEqual({ latMin: 10, latMax: 15, lonMin: 60, lonMax: 65 });
    expect(b).toEqual(a);
  });

  it("only ever grows the region and stays inside the domain", () => {
    const view = { latMin: 5, latMax: 6.3, lonMin: 104.2, lonMax: 105 };
    const plan = planRequestBounds(view);
    expect(boundsContain(plan, view)).toBe(true);
    expect(plan.latMin).toBeGreaterThanOrEqual(OCEAN_DOMAIN.latMin);
    expect(plan.lonMax).toBeLessThanOrEqual(OCEAN_DOMAIN.lonMax);
  });

  it("falls back to the exact viewport when snapping would exceed the backend cell limit", () => {
    // 63.9° x 63.9° at 0.25° is just under the 4096-cell limit exactly; snapping outward pushes it over
    const view = { latMin: 5.2, latMax: 20.9, lonMin: 45.2, lonMax: 60.9 };
    expect(estimateCells(view)).toBeLessThanOrEqual(MAX_GRID_POINTS);
    const snapped = { latMin: 5, latMax: 21, lonMin: 45, lonMax: 61 };
    expect(estimateCells(snapped)).toBeGreaterThan(MAX_GRID_POINTS);
    expect(planRequestBounds(view)).toEqual(view);
  });

  it("boundsContain reuses a loaded region for a zoom-in, not for a zoom-out or an offset view", () => {
    const loaded = { latMin: 10, latMax: 20, lonMin: 60, lonMax: 70 };
    expect(boundsContain(loaded, { latMin: 12, latMax: 14, lonMin: 62, lonMax: 66 })).toBe(true);
    expect(boundsContain(loaded, { latMin: 9, latMax: 14, lonMin: 62, lonMax: 66 })).toBe(false);
    expect(boundsContain(loaded, { latMin: 12, latMax: 14, lonMin: 65, lonMax: 71 })).toBe(false);
    expect(boundsContain(null, loaded)).toBe(false);
  });

  it("the full domain is over the cell limit (map asks the user to zoom rather than sending a doomed request)", () => {
    expect(
      estimateCells({ latMin: OCEAN_DOMAIN.latMin, latMax: OCEAN_DOMAIN.latMax, lonMin: OCEAN_DOMAIN.lonMin, lonMax: OCEAN_DOMAIN.lonMax })
    ).toBeGreaterThan(MAX_GRID_POINTS);
  });
});

describe("toGridSelection — click → grid cell", () => {
  it("snaps to the cell center and reports zero-based indices", () => {
    expect(toGridSelection(12.13, 80.62)).toEqual({ lat: 12.25, lon: 80.5, cell: { latIndex: 29, lonIndex: 142 } });
  });

  it("the domain corners map to the first and last cell", () => {
    expect(toGridSelection(5, 45).cell).toEqual({ latIndex: 0, lonIndex: 0 });
    expect(toGridSelection(30, 105).cell).toEqual({ latIndex: 100, lonIndex: 240 });
  });

  it("keeps out-of-domain clicks selectable but with no cell", () => {
    expect(toGridSelection(40.123, 10.987)).toEqual({ lat: 40.12, lon: 10.99, cell: null });
  });

  it("rejects invalid coordinates", () => {
    expect(toGridSelection(NaN, 60)).toBeNull();
    expect(toGridSelection(10, Infinity)).toBeNull();
    expect(toGridSelection(undefined, undefined)).toBeNull();
  });
});

describe("date/depth-driven map requests (Phase 36B)", () => {
  const region = { latMin: 10, latMax: 20, lonMin: 60, lonMax: 70 };

  it("only a real, non-negative depth is requestable — null must never mean 'all depths'", () => {
    expect(isRequestableDepth(0)).toBe(true);
    expect(isRequestableDepth(1000)).toBe(true);
    expect(isRequestableDepth(null)).toBe(false);
    expect(isRequestableDepth(undefined)).toBe(false);
    expect(isRequestableDepth(NaN)).toBe(false);
    expect(isRequestableDepth(-5)).toBe(false);
  });

  it("a change of date OR depth OR region is a different request; identical inputs are the same one", () => {
    const base = gridCacheKey({ date: "2020-01-01", depth: 10, ...region });
    expect(gridCacheKey({ date: "2020-01-01", depth: 10, ...region })).toBe(base);
    expect(gridCacheKey({ date: "2020-01-02", depth: 10, ...region })).not.toBe(base);
    expect(gridCacheKey({ date: "2020-01-01", depth: 20, ...region })).not.toBe(base);
    expect(gridCacheKey({ date: "2020-01-01", depth: 10, ...region, lonMax: 71 })).not.toBe(base);
  });

  it("a held field is 'current' only for exactly its own date and depth", () => {
    const ctx = { date: "2020-01-01", depth: 10 };
    expect(fieldContextMatches(ctx, { date: "2020-01-01", depth: 10 })).toBe(true);
    expect(fieldContextMatches(ctx, { date: "2020-01-02", depth: 10 })).toBe(false);
    expect(fieldContextMatches(ctx, { date: "2020-01-01", depth: 0 })).toBe(false);
    expect(fieldContextMatches(null, { date: "2020-01-01", depth: 10 })).toBe(false);
  });

  it("the response cache is a bounded LRU (revisiting a date/depth is free, memory is capped)", () => {
    const cache = new Map();
    for (let i = 0; i < GRID_CACHE_LIMIT + 3; i += 1) rememberInCache(cache, `k${i}`, i);
    expect(cache.size).toBe(GRID_CACHE_LIMIT);
    expect(cache.has("k0")).toBe(false);
    expect(cache.has(`k${GRID_CACHE_LIMIT + 2}`)).toBe(true);
    // touching an entry refreshes it so it outlives newer-but-untouched ones
    rememberInCache(cache, "k3", 3);
    rememberInCache(cache, "fresh", 1);
    expect(cache.has("k3")).toBe(true);
    expect(cache.has("k4")).toBe(false);
  });
});
