// -----------------------------------------------------------------------------
// NEER OceanMap — schematic landmask + projection helpers  (Phase 34C)
//
// The map draws an equirectangular projection of the domain (a reasonable
// simplification this close to the equator, and standard for regional
// oceanographic grids). Coastlines below are hand-simplified from well-known
// coastline shapes (Arabia/Iran/Pakistan, the Indian subcontinent, Myanmar,
// Sri Lanka, the Andaman & Nicobar chain) — they exist to orient the viewer
// and to mask land out of the reconstructed-ocean domain, NOT as a
// navigation-grade coastline. Nothing here encodes reconstructed field
// values; see OceanMap.js for the "demo data only" boundary.
// -----------------------------------------------------------------------------

import { OCEAN_DOMAIN } from "@/lib/oceanDomain";

/** SVG viewBox the map is drawn in — 16 user-units per degree on both axes. */
export const PX_PER_DEGREE = 16;
export const VIEW_BOX = Object.freeze({
  width: (OCEAN_DOMAIN.lonMax - OCEAN_DOMAIN.lonMin) * PX_PER_DEGREE,
  height: (OCEAN_DOMAIN.latMax - OCEAN_DOMAIN.latMin) * PX_PER_DEGREE,
});

/** Project a lon/lat pair to SVG (x, y) user-space coordinates. */
export function project([lon, lat], domain = OCEAN_DOMAIN) {
  const x = (lon - domain.lonMin) * PX_PER_DEGREE;
  const y = (domain.latMax - lat) * PX_PER_DEGREE;
  return [x, y];
}

/** Inverse of `project` — SVG (x, y) back to a lon/lat pair. */
export function unproject(x, y, domain = OCEAN_DOMAIN) {
  const lon = domain.lonMin + x / PX_PER_DEGREE;
  const lat = domain.latMax - y / PX_PER_DEGREE;
  return { lat, lon };
}

/** Build an SVG `points` attribute string from an array of [lon, lat] pairs. */
export function toSvgPoints(coords) {
  return coords.map((coord) => project(coord).join(",")).join(" ");
}

// Each entry is a closed ring of [lon, lat] vertices, simplified to the
// handful of points needed to read clearly at map scale.
export const LANDMASSES = [
  {
    id: "arabia-iran-pakistan-india-myanmar",
    name: "Arabia / Iran / Pakistan / Indian subcontinent / Myanmar",
    points: [
      [45, 30],
      [48, 29],
      [52, 27.5],
      [56, 26.5],
      [58, 25.5],
      [61, 24.8],
      [64, 24.5],
      [66, 24.2],
      [68, 23.8],
      [69, 22.5],
      [70, 21],
      [72.8, 19],
      [74, 15.5],
      [75.5, 11.5],
      [77, 8.2],
      [79.5, 9],
      [80, 13],
      [82, 17],
      [84, 19.5],
      [86.5, 21.5],
      [89, 22],
      [92, 21],
      [94, 18],
      [96, 16],
      [98, 14],
      [100, 12],
      [102, 10],
      [104, 9],
      [105, 9],
      [105, 30],
    ],
  },
  {
    id: "somalia",
    name: "Horn of Africa (Somalia)",
    points: [
      [45, 5],
      [46, 6],
      [47.5, 8],
      [49, 9.5],
      [51, 10.5],
      [51.5, 11.8],
      [49, 12],
      [46, 11.2],
      [45, 10],
    ],
  },
  {
    id: "sri-lanka",
    name: "Sri Lanka",
    points: [
      [79.8, 9.8],
      [81.2, 9.5],
      [81.9, 8.5],
      [81.5, 6.5],
      [80.3, 5.9],
      [79.8, 7],
    ],
  },
  {
    id: "andaman-nicobar",
    name: "Andaman & Nicobar Islands",
    points: [
      [92.5, 14],
      [93, 13],
      [93.2, 11],
      [93, 9],
      [92.8, 7],
      [92.6, 6.5],
      [93.2, 6.8],
      [93.5, 9],
      [93.7, 11.5],
      [93.5, 13.5],
    ],
  },
];

/**
 * Point-in-polygon test (ray casting) against a [lon, lat] ring.
 * Used to tell whether a hovered/selected coordinate falls on land, so the
 * map can flag it as outside the reconstructed-ocean domain.
 */
function pointInRing(lon, lat, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [loni, lati] = ring[i];
    const [lonj, latj] = ring[j];
    const intersects =
      lati > lat !== latj > lat &&
      lon < ((lonj - loni) * (lat - lati)) / (latj - lati) + loni;
    if (intersects) inside = !inside;
  }
  return inside;
}

/** True if a lon/lat coordinate falls within any schematic landmass ring. */
export function isOnLand(lat, lon) {
  return LANDMASSES.some((mass) => pointInRing(lon, lat, mass.points));
}