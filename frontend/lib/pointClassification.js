// -----------------------------------------------------------------------------
// NEER — point-location classification  (Phase 35C-B)
//
// A single, framework-independent place to answer one question: "given a
// lat/lon, is this a real NEER ocean grid cell we're allowed to query the
// reconstruction backend for, or not — and if not, why?" OceanMap.js (hover
// tooltip, click handling, coordinate readout) and
// dashboard/_context/PointInspectionContext.js (whether to call
// `reconstruct()` at all) both need the exact same answer for the exact same
// point; before this phase each one grew its own ad hoc
// isOnLand/in-domain check, which is exactly the kind of drift Requirement
// 10 elsewhere in this project (single source of truth, no duplicated
// logic) warns against.
//
// Deliberately takes `isOnLand` as a parameter rather than importing
// components/ocean-map/landmask.js directly: that file also carries the SVG
// projection helpers (project/unproject/PX_PER_DEGREE) and is a Client
// Component-adjacent module, whereas this one has no React/DOM dependency
// (same convention as lib/oceanDomain.js) and is exercised directly by
// vitest under lib/**. Callers (OceanMap.js, PointInspectionContext.js) each
// already have their own real `isOnLand` to pass in.
//
// Backend note (why this is a *client-side* classification, not a backend
// field): GET /reconstruct's model has no notion of land — it always
// returns a real number for any in-domain coordinate (see
// backend/app/services/repository.py's `reconstruct_point` /
// `src/inference/service.py`'s `predict_point` — neither consults a land
// mask). A "temperature" the model produces for a coordinate that's
// actually land is not a real ocean measurement, so the UI must not fetch
// or display one — it has to know the point is land *before* asking the
// backend, which is exactly what the schematic coastline in landmask.js is
// for (see that file's own header comment on why it's schematic, not
// navigation-grade).
// -----------------------------------------------------------------------------

import { OCEAN_DOMAIN, isWithinDomain } from "./oceanDomain";

/** The three ways a clicked/hovered point can relate to the NEER domain. */
export const POINT_STATUS = Object.freeze({
  /** Inside the NEER domain and not on land — safe to query the backend. */
  OCEAN: "ocean",
  /** Inside the NEER domain but on the schematic coastline/landmass. */
  LAND: "land",
  /** Outside the NEER domain (lat/lon bounds) entirely. */
  OUTSIDE_DOMAIN: "outside_domain",
});

/**
 * Classifies a lat/lon point as OCEAN, LAND, or OUTSIDE_DOMAIN.
 *
 * Domain bounds are checked first: a point outside the NEER domain is
 * OUTSIDE_DOMAIN regardless of whether it also happens to fall inside a
 * landmass ring (the landmass rings are only meaningful, and only need to
 * be evaluated, inside the domain).
 *
 * @param {number} lat
 * @param {number} lon
 * @param {(lat: number, lon: number) => boolean} isOnLand - e.g.
 *   components/ocean-map/landmask.js's `isOnLand`
 * @param {{latMin:number, latMax:number, lonMin:number, lonMax:number}} [domain=OCEAN_DOMAIN]
 * @returns {"ocean"|"land"|"outside_domain"}
 */
export function classifyPointLocation(lat, lon, isOnLand, domain = OCEAN_DOMAIN) {
  if (!isWithinDomain(lat, lon, domain)) return POINT_STATUS.OUTSIDE_DOMAIN;
  if (typeof isOnLand === "function" && isOnLand(lat, lon)) return POINT_STATUS.LAND;
  return POINT_STATUS.OCEAN;
}

/** True only for a point it's scientifically meaningful to reconstruct. */
export function isQueryablePoint(status) {
  return status === POINT_STATUS.OCEAN;
}

/**
 * Short, human-readable reason a non-OCEAN point can't be reconstructed —
 * shared wording for the map tooltip, the coordinate readout badge, and the
 * PointInspection/DashboardKPIs "can't query this point" states, so the
 * three surfaces never phrase the same fact three different ways.
 * @param {"ocean"|"land"|"outside_domain"} status
 * @returns {string|null} null for OCEAN (nothing to explain)
 */
export function describePointStatus(status) {
  switch (status) {
    case POINT_STATUS.LAND:
      return "Land — outside the reconstructed ocean domain";
    case POINT_STATUS.OUTSIDE_DOMAIN:
      return "Outside the NEER domain (5°N–30°N, 45°E–105°E)";
    default:
      return null;
  }
}

export default classifyPointLocation;