// -----------------------------------------------------------------------------
// NEER OceanMap — TemperatureLayer  (Phase 35B)
//
// Renders a backend-provided gridded scientific field (temperature or
// anomaly, from useOceanMapLayer.js / GET /reconstruct/grid) as a layer of
// colored grid cells, in the same world-space SVG coordinate system
// OceanMap.js's base map (ocean/land/graticule) is drawn in — see
// landmask.js's `project()`. This component is purely presentational: it
// takes already-fetched `grid`/`values` and a color mapper as props and
// draws them; it never calls the API itself (that stays in
// useOceanMapLayer.js) and never invents a value for a cell the backend
// didn't provide.
//
// Kept as its own component (rather than inlined into OceanMap.js) so it:
//   - can be shown/hidden independently of the base map (layer-visibility
//     requirement) without touching base-map drawing code,
//   - is reusable by any future map/variable that has a gridded backend
//     field (Phase 35A's LAYER_SUPPORT already anticipates sss/currents/
//     ssh/winds/subsurface/uncertainty grids being added later),
//   - keeps OceanMap.js's own render function from growing a second
//     responsibility.
//
// Missing/invalid cells: the backend serializes a non-finite value (masked
// land, no data for that cell) as JSON `null` (see backend
// `json_safe()` / src/data/preprocessing/_utils.py). Any such cell here is
// simply not drawn — left transparent, showing the base map underneath —
// rather than colored using a placeholder or interpolated value.
// -----------------------------------------------------------------------------

import { memo, useMemo } from "react";
import { PX_PER_DEGREE, project } from "./landmask";

/**
 * @param {object} props
 * @param {{lat: number[], lon: number[]}} props.grid - coordinate axes from
 *   the backend response (GridReconstructionResponse.lat/.lon)
 * @param {number[][]} props.values - `(lat.length, lon.length)` field
 *   values, already resolved to the active variable (temperature or the
 *   canonical backend anomaly) and with missing cells as
 *   `null` — see useOceanMapLayer.js
 * @param {(value: number|null) => (string|null)} props.colorMapper - maps
 *   one cell's value to a CSS color, or `null` to skip drawing that cell
 * @param {boolean} [props.visible=true] - layer-visibility toggle; when
 *   false, nothing is rendered (kept as a prop rather than unmounting the
 *   parent so toggling it back on doesn't re-trigger a fetch)
 * @param {number} [props.opacity=0.85]
 */
function TemperatureLayer({ grid, values, colorMapper, visible = true, opacity = 0.85 }) {
  const cells = useMemo(() => {
    const empty = { rects: [], cellWidthPx: 0, cellHeightPx: 0 };
    if (!visible || !grid || !values || !colorMapper) return empty;

    const { lat, lon } = grid;
    if (!lat?.length || !lon?.length) return empty;

    // Cell size from the grid's own spacing (never a hard-coded resolution)
    // — falls back to one degree only in the degenerate single-row/column
    // case, where there's no neighbouring sample to measure spacing from.
    const cellLatDeg = lat.length > 1 ? Math.abs(lat[1] - lat[0]) : 1;
    const cellLonDeg = lon.length > 1 ? Math.abs(lon[1] - lon[0]) : 1;
    const cellWidthPx = cellLonDeg * PX_PER_DEGREE;
    const cellHeightPx = cellLatDeg * PX_PER_DEGREE;

    const rects = [];
    for (let i = 0; i < lat.length; i += 1) {
      const row = values[i];
      if (!row) continue;
      for (let j = 0; j < lon.length; j += 1) {
        const value = row[j];
        const color = colorMapper(value);
        if (!color) continue; // missing/invalid cell — leave transparent
        const [cx, cy] = project([lon[j], lat[i]]);
        rects.push({
          key: `${i}-${j}`,
          x: cx - cellWidthPx / 2,
          y: cy - cellHeightPx / 2,
          color,
        });
      }
    }
    return { rects, cellWidthPx, cellHeightPx };
  }, [visible, grid, values, colorMapper]);

  if (!visible || !cells.rects.length) return null;

  return (
    <g data-layer="temperature-field" opacity={opacity}>
      {cells.rects.map((cell) => (
        <rect
          key={cell.key}
          x={cell.x}
          y={cell.y}
          width={cells.cellWidthPx}
          height={cells.cellHeightPx}
          fill={cell.color}
          stroke="none"
        />
      ))}
    </g>
  );
}

export default memo(TemperatureLayer);
