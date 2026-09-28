import { describe, expect, it } from "vitest";
import {
  POINT_STATUS,
  classifyPointLocation,
  describePointStatus,
  isQueryablePoint,
} from "./pointClassification";

const DOMAIN = { latMin: 5, latMax: 30, lonMin: 45, lonMax: 105 };

// A trivial land predicate: "on land" iff lon is an exact integer >= 80,
// just so tests can exercise the land branch without importing the real
// (much larger) landmass ring data from components/ocean-map/landmask.js.
const isOnLandStub = (lat, lon) => lon >= 80 && Number.isInteger(lon);

describe("classifyPointLocation", () => {
  it("classifies a mid-domain, non-land point as ocean", () => {
    expect(classifyPointLocation(15, 70.5, isOnLandStub, DOMAIN)).toBe(POINT_STATUS.OCEAN);
  });

  it("classifies a point outside the domain's latitude range as outside_domain", () => {
    expect(classifyPointLocation(40, 70, isOnLandStub, DOMAIN)).toBe(POINT_STATUS.OUTSIDE_DOMAIN);
  });

  it("classifies a point outside the domain's longitude range as outside_domain", () => {
    expect(classifyPointLocation(15, 10, isOnLandStub, DOMAIN)).toBe(POINT_STATUS.OUTSIDE_DOMAIN);
  });

  it("classifies an in-domain point flagged by isOnLand as land", () => {
    expect(classifyPointLocation(20, 80, isOnLandStub, DOMAIN)).toBe(POINT_STATUS.LAND);
  });

  it("treats domain bounds as inclusive", () => {
    expect(classifyPointLocation(5, 45, isOnLandStub, DOMAIN)).toBe(POINT_STATUS.OCEAN);
    expect(classifyPointLocation(30, 105, isOnLandStub, DOMAIN)).toBe(POINT_STATUS.OCEAN);
  });

  it("checks domain bounds before consulting isOnLand — an out-of-domain point never calls it", () => {
    let called = false;
    const spy = () => {
      called = true;
      return true;
    };
    expect(classifyPointLocation(1, 1, spy, DOMAIN)).toBe(POINT_STATUS.OUTSIDE_DOMAIN);
    expect(called).toBe(false);
  });

  it("defaults to OCEAN_DOMAIN when no domain is passed", () => {
    // 45°E-105°E, 5°N-30°N is the real NEER domain default.
    expect(classifyPointLocation(50, 200, () => false)).toBe(POINT_STATUS.OUTSIDE_DOMAIN);
  });

  it("tolerates a missing isOnLand function rather than throwing", () => {
    expect(classifyPointLocation(15, 70, undefined, DOMAIN)).toBe(POINT_STATUS.OCEAN);
  });
});

describe("isQueryablePoint", () => {
  it("is true only for ocean", () => {
    expect(isQueryablePoint(POINT_STATUS.OCEAN)).toBe(true);
    expect(isQueryablePoint(POINT_STATUS.LAND)).toBe(false);
    expect(isQueryablePoint(POINT_STATUS.OUTSIDE_DOMAIN)).toBe(false);
    expect(isQueryablePoint(undefined)).toBe(false);
  });
});

describe("describePointStatus", () => {
  it("returns null for ocean (nothing to explain)", () => {
    expect(describePointStatus(POINT_STATUS.OCEAN)).toBeNull();
  });

  it("returns a distinct, human-readable reason for land and outside_domain", () => {
    const land = describePointStatus(POINT_STATUS.LAND);
    const outside = describePointStatus(POINT_STATUS.OUTSIDE_DOMAIN);
    expect(land).toMatch(/land/i);
    expect(outside).toMatch(/outside/i);
    expect(land).not.toBe(outside);
  });
});