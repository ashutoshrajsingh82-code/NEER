// -----------------------------------------------------------------------------
// NEER OceanMap — GridOverlay  (Phase 35C-A)
//
// Draws the real NEER scientific reconstruction grid — 0.25° x 0.25°,
// aligned to the domain in lib/oceanDomain.js (single source of truth,
// mirrors configs/base.yaml and backend/app/services/repository.py's own
// `_grid_slice`) — as actual lon/lat grid-line geometry, not a generic
// repeating pixel pattern. Every line drawn here sits at an exact multiple
// of OCEAN_DOMAIN.resolution measured from the domain's own lat/lon
// origin, so it always lines up with the real grid cells GET
// /reconstruct/grid returns (TemperatureLayer.js draws those cells from
// the same lat/lon axes) — this overlay never invents its own spacing or
// a decorative texture unrelated to the actual spatial resolution.
//
// Purely presentational, same contract as TemperatureLayer.js: given
// `bounds`/`scale`/`visible` it draws lines and touches the network for
// nothing. Two things keep it "integrated" rather than "decorative":
//   - its spacing is derived from OCEAN_DOMAIN.resolution, so a config
//     change to the NEER grid resolution changes what's drawn here too;
//   - it only builds lines inside the visible viewport (`bounds`, already
//     clamped to OCEAN_DOMAIN by OceanMap.js) and coarsens its stride as
//     the user zooms out (see `strideFor`), rather than always laying
//     down the full ~240x100 line domain regardless of what's legible or
//     what's actually on screen.
//
// Drawn above TemperatureLayer/anomaly field but below the bolder 5°
// graticule/coastline/domain-boundary chrome, at a low, fixed opacity, so
// the scientific field underneath is never obscured — the grid is a
// reference, not a competing visual layer.
// -----------------------------------------------------------------------------

import { memo, useMemo } from "react";
import { OCEAN_DOMAIN, clamp } from "@/lib/oceanDomain";
import { PX_PER_DEGREE, project } from "./landmask";

/** Below this zoom, 0.25° cells are too small to read meaningfully — the
 * grid simply doesn't render rather than becoming visual noise. */
export const GRID_MIN_VISIBLE_SCALE = 1.75;

/**
 * How many resolution-steps apart drawn lines are, coarsening as the user
 * zooms out so the overlay never has to build/draw the domain's full line
 * count when only a handful would be legible on screen.
 */
function strideFor(scale) {
  if (scale >= 6) return 1; // native 0.25°
  if (scale >= 3.5) return 2; // 0.5°
  return 4; // 1°
}

/**
 * @param {object} props
 * @param {{latMin:number, latMax:number, lonMin:number, lonMax:number}} props.bounds
 *   - visible viewport, already clamped to OCEAN_DOMAIN (OceanMap.js's
 *     `visibleBounds`) — only lines inside this region are built.
 * @param {number} props.scale - current map zoom (transform.scale)
 * @param {boolean} [props.visible=true] - layer-visibility toggle
 */
function GridOverlay({ bounds, scale, visible = true }) {
  const { resolution, latMin: domainLatMin, latMax: domainLatMax, lonMin: domainLonMin, lonMax: domainLonMax } =
    OCEAN_DOMAIN;

  const stride = strideFor(scale);
  const step = resolution * stride;

  const { latLines, lonLines } = useMemo(() => {
    if (!visible || !bounds || scale < GRID_MIN_VISIBLE_SCALE) return { latLines: [], lonLines: [] };

    const visibleLatMin = clamp(bounds.latMin, domainLatMin, domainLatMax);
    const visibleLatMax = clamp(bounds.latMax, domainLatMin, domainLatMax);
    const visibleLonMin = clamp(bounds.lonMin, domainLonMin, domainLonMax);
    const visibleLonMax = clamp(bounds.lonMax, domainLonMin, domainLonMax);

    const latLines = [];
    const latStart = domainLatMin + Math.floor((visibleLatMin - domainLatMin) / step) * step;
    for (let lat = latStart; lat <= visibleLatMax + 1e-9; lat += step) {
      latLines.push(Math.round(lat * 1000) / 1000);
    }

    const lonLines = [];
    const lonStart = domainLonMin + Math.floor((visibleLonMin - domainLonMin) / step) * step;
    for (let lon = lonStart; lon <= visibleLonMax + 1e-9; lon += step) {
      lonLines.push(Math.round(lon * 1000) / 1000);
    }

    return { latLines, lonLines };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, scale, step, bounds?.latMin, bounds?.latMax, bounds?.lonMin, bounds?.lonMax]);

  if (!visible || (latLines.length === 0 && lonLines.length === 0)) return null;

  const domainWidthPx = (domainLonMax - domainLonMin) * PX_PER_DEGREE;
  const domainHeightPx = (domainLatMax - domainLatMin) * PX_PER_DEGREE;
  const strokeWidth = 0.35 / Math.max(scale, 1);

  return (
    <g data-layer="scientific-grid" aria-hidden="true">
      {latLines.map((lat) => {
        const [, y] = project([domainLonMin, lat]);
        return (
          <line
            key={`grid-lat-${lat}`}
            x1={0}
            y1={y}
            x2={domainWidthPx}
            y2={y}
            stroke="rgba(148,197,224,0.16)"
            strokeWidth={strokeWidth}
          />
        );
      })}
      {lonLines.map((lon) => {
        const [x] = project([lon, domainLatMax]);
        return (
          <line
            key={`grid-lon-${lon}`}
            x1={x}
            y1={0}
            x2={x}
            y2={domainHeightPx}
            stroke="rgba(148,197,224,0.16)"
            strokeWidth={strokeWidth}
          />
        );
      })}
    </g>
  );
}

export default memo(GridOverlay);