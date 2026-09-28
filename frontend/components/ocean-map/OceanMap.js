"use client";

// -----------------------------------------------------------------------------
// NEER — OceanMap  (Phase 34C)
//
// The main interactive map region of /dashboard: a custom SVG scientific
// visualization of the North Indian Ocean reconstruction domain (5°N–30°N,
// 45°E–105°E at 0.25° × 0.25°) — pan, zoom, hover, and grid-cell selection,
// built from scratch rather than a generic tiled web-map library (none is
// present in the project; see landmask.js for why an equirectangular custom
// SVG projection fits this domain and this app's dark scientific-instrument
// visual language better than a basemap-tile library would).
//
// Phase 34C scope: layout, projection, interaction (pan/zoom/select/hover),
// and visual chrome (ocean/land/domain-boundary/selection distinction, a
// compact controls area, and date/depth/mode context) are real and
// functional. No reconstructed field is wired in yet — the variable tabs and
// legend are scaffolding for later phases to attach real layers to; nothing
// here invents a temperature, salinity, or other scientific value. Point
// selection is local UI state (with an `onSelectPoint` escape hatch) — wiring
// it to the shared InspectionPanel is left to whichever later phase owns
// that integration.
//
// Phase 35A adds the "architecture ... for backend-driven scientific
// layers" its spec asks for, via useOceanMapLayer.js: a real GET
// /reconstruct/grid fetch, keyed off the active variable tab / date / depth
// / visible viewport — see that file's header for the variable ->
// backend-support table and the cell-limit check.
//
// Phase 35B renders that fetch: `layer.values`/`layer.grid` are drawn as a
// real, separate TemperatureLayer.js SVG layer (own visibility toggle, own
// <g>, composited over the base map but under the coastline mask so
// land — including any cell the backend didn't mask itself — never shows a
// fabricated ocean color), colored with a continuous scale from
// lib/colorScale.js domain-fitted to that field's own real min/max (never a
// hard-coded physical range). Loading/error/empty states for the fetch are
// surfaced here as their own overlays — a LoadingSkeleton pill, ErrorState
// with retry, and a plain "no field" message — rather than silently leaving
// the map blank. Nothing here still invents a scientific value: an
// unsupported variable (sss/currents/ssh/winds/subsurface/uncertainty) or a
// cell the backend returned as `null` never gets drawn.
//
// Phase 35C-A adds the temperature-anomaly and scientific-grid layers this
// phase's spec asks for:
//   - Anomaly is `activeVariable === "anomaly"`, already served by the same
//     useOceanMapLayer.js/TemperatureLayer.js machinery as SST (see that
//     hook's header for exactly how `temperature - climatology` is derived
//     from two real backend grids, never invented) — this phase makes its
//     loading/error/missing-data states share the same StatusIndicator
//     vocabulary as the rest of the design system instead of ad hoc markup.
//   - GridOverlay.js replaces the old fixed-pixel "fineGrid" pattern with
//     real 0.25°-aligned grid-line geometry, generated from
//     OCEAN_DOMAIN.resolution (the same constant TemperatureLayer's cells
//     and every /reconstruct/grid request already use), clipped to the
//     visible viewport and coarsened by zoom so it never obscures the
//     temperature/anomaly field beneath it.
// -----------------------------------------------------------------------------

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  ArrowUpDown,
  CalendarDays,
  Compass,
  Database,
  Droplets,
  Layers,
  LayoutGrid,
  Maximize2,
  Minimize2,
  MoveVertical,
  Navigation,
  Thermometer,
  TrendingUp,
  RotateCcw,
  Waves,
  Wind,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { Badge, Button, ErrorState, LoadingSkeleton, Panel, StatusIndicator, Tabs, Tooltip } from "@/components/ui";
import { cn } from "@/lib/cn";
import { formatSigned } from "@/lib/format";
import {
  VARIABLE_COLOR_CONFIG,
  computeFiniteExtent,
  divergingColor,
  makeColorMapper,
  thermalColor,
} from "@/lib/colorScale";
import {
  OCEAN_DOMAIN,
  clamp,
  formatDepth,
  formatLat,
  formatLon,
  isWithinDomain,
  round2,
  snapToGrid,
} from "@/lib/oceanDomain";
import { POINT_STATUS, classifyPointLocation, describePointStatus } from "@/lib/pointClassification";
import CoordinateDisplay from "./CoordinateDisplay";
import GridOverlay from "./GridOverlay";
import { LANDMASSES, VIEW_BOX, isOnLand, project, unproject } from "./landmask";
import MapControls from "./MapControls";
import MapLegend from "./MapLegend";
import TemperatureLayer from "./TemperatureLayer";
import { useOceanMapLayer } from "./useOceanMapLayer";

// Variable/layer tabs — the four with real interaction wired (sst/sss/
// currents/anomaly) plus the remaining layers named in the phase spec,
// present but disabled so the map's chrome is already shaped for them.
const VARIABLE_TABS = [
  { value: "sst", label: "Sea Surface Temp.", icon: Thermometer },
  { value: "sss", label: "Salinity", icon: Droplets },
  { value: "currents", label: "Currents", icon: Navigation },
  { value: "anomaly", label: "Anomaly", icon: TrendingUp },
  { value: "ssh", label: "SSH / SLA", icon: ArrowUpDown, disabled: true },
  { value: "winds", label: "Winds", icon: Wind, disabled: true },
  { value: "subsurface", label: "Subsurface Temp.", icon: Layers, disabled: true },
  { value: "uncertainty", label: "Uncertainty", icon: AlertTriangle, disabled: true },
];

const MODE_CONFIG = {
  reconstructed: { label: "Reconstructed", variant: "accent" },
  observed: { label: "Observed", variant: "info" },
  blended: { label: "Blended", variant: "warning" },
};

const MIN_SCALE = 1;
const MAX_SCALE = 8;
const ZOOM_STEP = 1.35;
const DATE_FORMAT = { day: "2-digit", month: "short", year: "numeric" };

/** Pan bounds so the domain can never be dragged entirely out of view. */
function clampPan(scale, x, y) {
  const minX = VIEW_BOX.width * (1 - scale);
  const minY = VIEW_BOX.height * (1 - scale);
  return { x: clamp(x, minX, 0), y: clamp(y, minY, 0) };
}

/**
 * Maps a client (mouse/touch) pixel position to viewBox units, accounting for
 * the SVG's `preserveAspectRatio="xMidYMid meet"` letterboxing — the SVG
 * content is centered and scaled to fit the container without distortion,
 * so a naive width-fraction mapping would be wrong whenever the container's
 * aspect ratio doesn't exactly match the viewBox's.
 */
function clientToViewBox(clientX, clientY, rect) {
  const containerAspect = rect.width / rect.height;
  const viewBoxAspect = VIEW_BOX.width / VIEW_BOX.height;
  let pxPerUnit;
  let offsetX = 0;
  let offsetY = 0;
  if (containerAspect > viewBoxAspect) {
    pxPerUnit = rect.height / VIEW_BOX.height;
    offsetX = (rect.width - VIEW_BOX.width * pxPerUnit) / 2;
  } else {
    pxPerUnit = rect.width / VIEW_BOX.width;
    offsetY = (rect.height - VIEW_BOX.height * pxPerUnit) / 2;
  }
  return {
    x: (clientX - rect.left - offsetX) / pxPerUnit,
    y: (clientY - rect.top - offsetY) / pxPerUnit,
  };
}

/**
 * Nearest-cell lookup of a real field value for the coordinate readout —
 * reads the same `grid`/`values` the map layer draws, never a separately
 * fetched or fabricated number. Returns `null` when there's no loaded grid,
 * the point falls outside it, or the nearest cell was itself missing/masked.
 */
function lookupFieldValue(grid, values, point) {
  if (!grid || !values || !point) return null;
  const { lat, lon } = grid;
  if (!lat?.length || !lon?.length) return null;
  const latIndex = lat.reduce(
    (best, candidate, i) => (Math.abs(candidate - point.lat) < Math.abs(lat[best] - point.lat) ? i : best),
    0
  );
  const lonIndex = lon.reduce(
    (best, candidate, i) => (Math.abs(candidate - point.lon) < Math.abs(lon[best] - point.lon) ? i : best),
    0
  );
  const value = values[latIndex]?.[lonIndex];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function ToolbarButton({ icon, label, onClick, active }) {
  return (
    <Tooltip content={label}>
      <span>
        <Button
          variant={active ? "secondary" : "ghost"}
          size="sm"
          iconOnly
          icon={icon}
          aria-label={label}
          aria-pressed={active || undefined}
          onClick={onClick}
        />
      </span>
    </Tooltip>
  );
}

/**
 * @param {"reconstructed"|"observed"|"blended"} dataMode
 * @param {Date|string} date - date currently in view, display-only here
 * @param {number} depth - depth level in metres currently in view
 * @param {string} variable - controlled active layer tab; falls back to internal state
 * @param {(value: string) => void} onVariableChange
 * @param {{lat: number, lon: number}} selectedPoint - controlled selection; falls back to internal state
 * @param {(point: {lat: number, lon: number}) => void} onSelectPoint - called with the snapped grid-cell center on click
 */
export default function OceanMap({
  dataMode = "reconstructed",
  date,
  depth = 0,
  variable,
  onVariableChange,
  selectedPoint,
  onSelectPoint,
  className,
}) {
  const containerRef = useRef(null);
  const dragRef = useRef(null);

  const [internalVariable, setInternalVariable] = useState("sst");
  const [internalSelected, setInternalSelected] = useState(null);
  const [transform, setTransform] = useState({ scale: 1, x: 0, y: 0 });
  const [hover, setHover] = useState(null); // {lat, lon} in world space, or null
  // Container-relative pixel position of the same hover, purely so the
  // Phase 35C-B floating scientific tooltip can sit next to the cursor —
  // kept separate from `hover` (world lat/lon) because the tooltip's CSS
  // `left`/`top` need real pixels, not viewBox units, and re-deriving pixels
  // from `hover` would have to re-invert the same letterboxing math
  // `clientToViewBox` already did for this exact event.
  const [hoverScreenPos, setHoverScreenPos] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [layerVisibility, setLayerVisibility] = useState({
    graticule: true,
    scientificGrid: true,
    coastline: true,
    domainBoundary: true,
    temperatureField: true,
  });

  const isVariableControlled = variable !== undefined;
  const activeVariable = isVariableControlled ? variable : internalVariable;
  const isSelectionControlled = selectedPoint !== undefined;
  const activeSelected = isSelectionControlled ? selectedPoint : internalSelected;
  const activeDate = date ? new Date(date) : null;
  const modeConfig = MODE_CONFIG[dataMode] ?? MODE_CONFIG.reconstructed;
  const activeLabel = VARIABLE_TABS.find((tab) => tab.value === activeVariable)?.label;

  function changeVariable(next) {
    if (!isVariableControlled) setInternalVariable(next);
    onVariableChange?.(next);
  }

  const selectPoint = useCallback(
    (point) => {
      if (!isSelectionControlled) setInternalSelected(point);
      onSelectPoint?.(point);
    },
    [isSelectionControlled, onSelectPoint]
  );

  /** World-space (lon, lat) → current screen position, in viewBox units. */
  const worldToScreen = useCallback(
    (lon, lat) => {
      const [wx, wy] = project([lon, lat]);
      return { x: transform.x + wx * transform.scale, y: transform.y + wy * transform.scale };
    },
    [transform]
  );

  const zoomAtViewBoxPoint = useCallback((rawX, rawY, factor) => {
    setTransform((prev) => {
      const newScale = clamp(prev.scale * factor, MIN_SCALE, MAX_SCALE);
      if (newScale === prev.scale) return prev;
      const worldX = (rawX - prev.x) / prev.scale;
      const worldY = (rawY - prev.y) / prev.scale;
      const { x, y } = clampPan(newScale, rawX - worldX * newScale, rawY - worldY * newScale);
      return { scale: newScale, x, y };
    });
  }, []);

  function zoomByButton(factor) {
    zoomAtViewBoxPoint(VIEW_BOX.width / 2, VIEW_BOX.height / 2, factor);
  }

  function resetView() {
    setTransform({ scale: 1, x: 0, y: 0 });
  }

  function handleWheel(event) {
    event.preventDefault();
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const raw = clientToViewBox(event.clientX, event.clientY, rect);
    zoomAtViewBoxPoint(raw.x, raw.y, event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP);
  }

  function handlePointerDown(event) {
    if (event.button !== 0 && event.pointerType === "mouse") return;
    containerRef.current?.setPointerCapture(event.pointerId);
    dragRef.current = {
      startClientX: event.clientX,
      startClientY: event.clientY,
      startX: transform.x,
      startY: transform.y,
      moved: false,
    };
  }

  function handlePointerMove(event) {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;

    if (dragRef.current) {
      const drag = dragRef.current;
      const dxClient = event.clientX - drag.startClientX;
      const dyClient = event.clientY - drag.startClientY;
      if (Math.abs(dxClient) + Math.abs(dyClient) > 3) drag.moved = true;

      const containerAspect = rect.width / rect.height;
      const viewBoxAspect = VIEW_BOX.width / VIEW_BOX.height;
      const pxPerUnit =
        containerAspect > viewBoxAspect ? rect.height / VIEW_BOX.height : rect.width / VIEW_BOX.width;

      const { x, y } = clampPan(
        transform.scale,
        drag.startX + dxClient / pxPerUnit,
        drag.startY + dyClient / pxPerUnit
      );
      setTransform((prev) => ({ ...prev, x, y }));
    }

    const raw = clientToViewBox(event.clientX, event.clientY, rect);
    const worldX = (raw.x - transform.x) / transform.scale;
    const worldY = (raw.y - transform.y) / transform.scale;
    setHover(unproject(worldX, worldY));
    setHoverScreenPos({ x: event.clientX - rect.left, y: event.clientY - rect.top });
  }

  function handlePointerUp(event) {
    const drag = dragRef.current;
    dragRef.current = null;
    containerRef.current?.releasePointerCapture?.(event.pointerId);
    if (!drag || drag.moved) return;

    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const raw = clientToViewBox(event.clientX, event.clientY, rect);
    const worldX = (raw.x - transform.x) / transform.scale;
    const worldY = (raw.y - transform.y) / transform.scale;
    const { lat, lon } = unproject(worldX, worldY);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
    if (isWithinDomain(lat, lon)) {
      selectPoint(snapToGrid(lat, lon));
    } else {
      selectPoint({ lat: round2(lat), lon: round2(lon) });
    }
  }

  function handleDoubleClick(event) {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const raw = clientToViewBox(event.clientX, event.clientY, rect);
    zoomAtViewBoxPoint(raw.x, raw.y, ZOOM_STEP * 1.3);
  }

  function handleKeyDown(event) {
    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      zoomByButton(ZOOM_STEP);
    } else if (event.key === "-" || event.key === "_") {
      event.preventDefault();
      zoomByButton(1 / ZOOM_STEP);
    } else if (event.key === "0") {
      event.preventDefault();
      resetView();
    } else if (event.key === "Escape") {
      if (activeSelected) {
        event.preventDefault();
        selectPoint(null);
      }
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setTransform((prev) => {
        const { x, y } = clampPan(prev.scale, prev.x, prev.y + 40);
        return { ...prev, x, y };
      });
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      setTransform((prev) => {
        const { x, y } = clampPan(prev.scale, prev.x, prev.y - 40);
        return { ...prev, x, y };
      });
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      setTransform((prev) => {
        const { x, y } = clampPan(prev.scale, prev.x + 40, prev.y);
        return { ...prev, x, y };
      });
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      setTransform((prev) => {
        const { x, y } = clampPan(prev.scale, prev.x - 40, prev.y);
        return { ...prev, x, y };
      });
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (hover && Number.isFinite(hover.lat) && Number.isFinite(hover.lon)) {
        if (isWithinDomain(hover.lat, hover.lon)) {
          selectPoint(snapToGrid(hover.lat, hover.lon));
        } else {
          selectPoint({ lat: round2(hover.lat), lon: round2(hover.lon) });
        }
      }
    }
  }

  // Fullscreen toggle for the map surface itself, with a graceful fallback
  // for browsers/environments without the Fullscreen API.
  useEffect(() => {
    function onFullscreenChange() {
      setIsFullscreen(document.fullscreenElement === containerRef.current);
    }
    document.addEventListener("fullscreenchange", onFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", onFullscreenChange);
  }, []);

  function toggleFullscreen() {
    if (!containerRef.current) return;
    if (document.fullscreenElement) {
      document.exitFullscreen?.();
    } else {
      containerRef.current.requestFullscreen?.().catch(() => {});
    }
  }

  function toggleLayer(key) {
    setLayerVisibility((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  // 5° graticule lines, drawn in world space so they pan/zoom with the map.
  const latLines = useMemo(() => {
    const lines = [];
    for (let lat = OCEAN_DOMAIN.latMin; lat <= OCEAN_DOMAIN.latMax; lat += 5) lines.push(lat);
    return lines;
  }, []);
  const lonLines = useMemo(() => {
    const lines = [];
    for (let lon = OCEAN_DOMAIN.lonMin; lon <= OCEAN_DOMAIN.lonMax; lon += 5) lines.push(lon);
    return lines;
  }, []);

  const graticuleLines = useMemo(() => {
    if (!layerVisibility.graticule) return null;
    return (
      <>
        {latLines.map((lat) => {
          const [, y] = project([OCEAN_DOMAIN.lonMin, lat]);
          return (
            <line
              key={`lat-${lat}`}
              x1={0}
              y1={y}
              x2={VIEW_BOX.width}
              y2={y}
              stroke="rgba(148,197,224,0.14)"
              strokeWidth={0.5 / transform.scale}
            />
          );
        })}
        {lonLines.map((lon) => {
          const [x] = project([lon, OCEAN_DOMAIN.latMax]);
          return (
            <line
              key={`lon-${lon}`}
              x1={x}
              y1={0}
              x2={x}
              y2={VIEW_BOX.height}
              stroke="rgba(148,197,224,0.14)"
              strokeWidth={0.5 / transform.scale}
            />
          );
        })}
      </>
    );
  }, [layerVisibility.graticule, latLines, lonLines, transform.scale]);

  const coastlineElements = useMemo(() => {
    if (!layerVisibility.coastline) return null;
    return LANDMASSES.map((mass) => (
      <polygon
        key={mass.id}
        points={mass.svgPoints ?? mass.points.map((p) => project(p).join(",")).join(" ")}
        fill="#132133"
        stroke="rgba(148,197,224,0.3)"
        strokeWidth={0.75 / transform.scale}
      />
    ));
  }, [layerVisibility.coastline, transform.scale]);

  const domainBoundaryElement = useMemo(() => {
    if (!layerVisibility.domainBoundary) return null;
    return (
      <rect
        x={0}
        y={0}
        width={VIEW_BOX.width}
        height={VIEW_BOX.height}
        fill="none"
        stroke="rgba(45,212,191,0.55)"
        strokeWidth={1.5 / transform.scale}
        strokeDasharray={`${6 / transform.scale} ${5 / transform.scale}`}
      />
    );
  }, [layerVisibility.domainBoundary, transform.scale]);

  // Current visible viewport, in lon/lat — same inverse-projection
  // handlePointerMove already uses per-pixel, applied to the surface's two
  // corners instead. This is the region useOceanMapLayer requests: panning/
  // zooming changes what's fetched, not just what's drawn.
  const visibleBounds = useMemo(() => {
    const topLeft = unproject((0 - transform.x) / transform.scale, (0 - transform.y) / transform.scale);
    const bottomRight = unproject(
      (VIEW_BOX.width - transform.x) / transform.scale,
      (VIEW_BOX.height - transform.y) / transform.scale
    );
    return {
      latMin: clamp(round2(bottomRight.lat), OCEAN_DOMAIN.latMin, OCEAN_DOMAIN.latMax),
      latMax: clamp(round2(topLeft.lat), OCEAN_DOMAIN.latMin, OCEAN_DOMAIN.latMax),
      lonMin: clamp(round2(topLeft.lon), OCEAN_DOMAIN.lonMin, OCEAN_DOMAIN.lonMax),
      lonMax: clamp(round2(bottomRight.lon), OCEAN_DOMAIN.lonMin, OCEAN_DOMAIN.lonMax),
    };
  }, [transform]);

  const layer = useOceanMapLayer({ variable: activeVariable, date, depth, bounds: visibleBounds });

  // Color scale for whatever field is actually loaded — domain-fitted to
  // that field's own real finite min/max (computeFiniteExtent), never a
  // hard-coded physical range, and `null` whenever there's nothing to draw
  // yet (unsupported variable, no data, or every cell masked).
  const colorConfig = VARIABLE_COLOR_CONFIG[activeVariable] ?? null;
  const fieldExtent = layer.values ? computeFiniteExtent(layer.values) : null;
  const colorMapper = colorConfig && fieldExtent ? makeColorMapper(colorConfig.kind, fieldExtent) : null;
  const legendSample = colorConfig?.kind === "diverging" ? divergingColor : thermalColor;

  // "No field available" is distinct from "loading"/"errored"/"unsupported":
  // the request succeeded but there is genuinely nothing to draw (e.g. no
  // climatology for this date, so anomaly can't be derived).
  const showEmptyFieldState =
    layer.supported && Boolean(date) && layer.status === "success" && !layer.isLoading && !layer.values;

  // Single source of truth for "is this point ocean, land, or outside the
  // NEER domain" (lib/pointClassification.js) — used for the hover tooltip,
  // the click/selection badge, and (crucially, in
  // PointInspectionContext.js) whether a click is even allowed to reach
  // GET /reconstruct. `hoverInDomain` keeps its own small ±2° grace margin
  // for the *crosshair marker's* visibility only (a purely cosmetic "don't
  // pop the crosshair in/out right at the domain edge" concern) — it is
  // never used to decide what data to show or fetch, which is why it stays
  // separate from the strict classification below.
  const hoverStatus = hover ? classifyPointLocation(hover.lat, hover.lon, isOnLand) : null;
  const selectedStatus = activeSelected
    ? classifyPointLocation(activeSelected.lat, activeSelected.lon, isOnLand)
    : null;
  const hoverOnLand = hoverStatus === POINT_STATUS.LAND;
  const selectedOnLand = selectedStatus === POINT_STATUS.LAND;
  const hoverInDomain =
    hover &&
    hover.lat >= OCEAN_DOMAIN.latMin - 2 &&
    hover.lat <= OCEAN_DOMAIN.latMax + 2 &&
    hover.lon >= OCEAN_DOMAIN.lonMin - 2 &&
    hover.lon <= OCEAN_DOMAIN.lonMax + 2;

  const readoutPoint = hover ?? activeSelected ?? null;
  const readoutStatus = hover ? hoverStatus : selectedStatus;
  const readoutValue = lookupFieldValue(layer.grid, layer.values, readoutPoint);
  const readoutValueText =
    readoutStatus && readoutStatus !== POINT_STATUS.OCEAN
      ? "N/A"
      : readoutValue === null
        ? "--"
        : `${readoutValue.toFixed(2)}${colorConfig?.unit ?? ""}`;

  // Hover tooltip's temperature/anomaly — read from the same already-loaded
  // grid the colored field/legend draw from (never a separate fetch, never
  // invented), looked up by nearest cell exactly like `readoutValue` above.
  // Unlike `readoutValue` (whichever single field the active tab is
  // showing), the tooltip always tries both, since Requirement 1's tooltip
  // spec asks for "temperature" and "anomaly where available" together —
  // see useOceanMapLayer.js's `temperatureValues`/`anomalyValues` for why
  // both are derivable from one fetched response regardless of which tab
  // (sst/anomaly) is active.
  const hoverTemperature =
    hoverStatus === POINT_STATUS.OCEAN ? lookupFieldValue(layer.grid, layer.temperatureValues, hover) : null;
  const hoverAnomaly =
    hoverStatus === POINT_STATUS.OCEAN ? lookupFieldValue(layer.grid, layer.anomalyValues, hover) : null;

  // MapControls' primary, spec-required toggles:
  // 1. Temperature layer visibility (turns SST on/off, or switches to SST)
  // 2. Temperature anomaly layer visibility (turns anomaly on/off, or switches to anomaly)
  // 3. Grid overlay visibility (0.25° scientific grid)
  const mapControlLayers = [
    {
      key: "temperature-field",
      icon: Thermometer,
      label: "Temperature layer",
      description: "Sea-surface temperature field",
      active: layerVisibility.temperatureField && activeVariable === "sst",
      onToggle: () => {
        if (layerVisibility.temperatureField && activeVariable === "sst") {
          toggleLayer("temperatureField");
        } else {
          changeVariable("sst");
          setLayerVisibility((prev) => ({ ...prev, temperatureField: true }));
        }
      },
    },
    {
      key: "anomaly-field",
      icon: TrendingUp,
      label: "Temperature anomaly layer",
      description: "Temperature anomaly relative to climatology",
      active: layerVisibility.temperatureField && activeVariable === "anomaly",
      onToggle: () => {
        if (layerVisibility.temperatureField && activeVariable === "anomaly") {
          toggleLayer("temperatureField");
        } else {
          changeVariable("anomaly");
          setLayerVisibility((prev) => ({ ...prev, temperatureField: true }));
        }
      },
    },
    {
      key: "scientific-grid",
      icon: LayoutGrid,
      label: "0.25° scientific grid",
      description: "NEER native 0.25° × 0.25° grid overlay",
      active: layerVisibility.scientificGrid,
      onToggle: () => toggleLayer("scientificGrid"),
    },
  ];

  // Earlier phases' base-map chrome — kept togglable, grouped separately so
  // the popover reads "the scientific layers you asked for" first and "the
  // rest of the map's chrome" second, rather than one flat undifferentiated
  // list.
  const mapBaseLayers = [
    { key: "graticule", icon: Compass, label: "Graticule & labels", active: layerVisibility.graticule },
    { key: "coastline", icon: Waves, label: "Coastline", active: layerVisibility.coastline },
    { key: "domainBoundary", icon: Database, label: "Domain boundary", active: layerVisibility.domainBoundary },
  ].map((item) => ({ ...item, onToggle: () => toggleLayer(item.key) }));

  return (
    <Panel
      emphasis="raised"
      title="Ocean Reconstruction Map"
      subtitle={`North Indian Ocean — ${OCEAN_DOMAIN.resolution}° gridded field`}
      icon={Waves}
      bodyClassName="flex flex-col gap-4"
      headerActions={
        <div className="hidden items-center gap-1 sm:flex">
          <ToolbarButton
            icon={isFullscreen ? Minimize2 : Maximize2}
            label={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
            onClick={toggleFullscreen}
          />
        </div>
      }
    >
      <Tabs
        items={VARIABLE_TABS}
        value={activeVariable}
        onChange={changeVariable}
        id="dashboard-map-variable"
        className="overflow-x-auto"
      />

      {/* Live context strip — what slice of the domain the map represents */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-caption">
        <span className="flex items-center gap-1.5 text-text-muted">
          <Database size={13} strokeWidth={1.75} aria-hidden="true" />
          <Badge variant={modeConfig.variant} size="sm">
            {modeConfig.label}
          </Badge>
        </span>
        <span className="flex items-center gap-1.5 text-text-muted">
          <CalendarDays size={13} strokeWidth={1.75} aria-hidden="true" />
          <span className="font-mono text-text-secondary">
            {activeDate ? activeDate.toLocaleDateString("en-GB", DATE_FORMAT) : "No date selected"}
          </span>
        </span>
        <span className="flex items-center gap-1.5 text-text-muted">
          <MoveVertical size={13} strokeWidth={1.75} aria-hidden="true" />
          <span className="font-mono text-text-secondary">{formatDepth(depth)}</span>
        </span>
      </div>

      {/* Map surface */}
      <div
        ref={containerRef}
        role="application"
        aria-label="Interactive North Indian Ocean grid map. Drag to pan, scroll to zoom, click to select a grid cell."
        tabIndex={0}
        onWheel={handleWheel}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={() => {
          setHover(null);
          setHoverScreenPos(null);
        }}
        onDoubleClick={handleDoubleClick}
        onKeyDown={handleKeyDown}
        className={cn(
          "relative min-h-[420px] flex-1 touch-none select-none overflow-hidden rounded-md border border-border",
          "bg-grid-subtle bg-grid focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400",
          "lg:min-h-[480px]",
          className
        )}
        style={{ cursor: dragRef.current ? "grabbing" : "grab" }}
      >
        <svg
          viewBox={`0 0 ${VIEW_BOX.width} ${VIEW_BOX.height}`}
          preserveAspectRatio="xMidYMid meet"
          className="absolute inset-0 h-full w-full"
        >
          <defs>
            <linearGradient id="oceanGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#0E2A42" />
              <stop offset="100%" stopColor="#081826" />
            </linearGradient>
          </defs>

          {/* Pannable/zoomable map content — world-space coordinates */}
          <g transform={`translate(${transform.x} ${transform.y}) scale(${transform.scale})`}>
            <rect x={0} y={0} width={VIEW_BOX.width} height={VIEW_BOX.height} fill="url(#oceanGradient)" />

            {/* Backend-driven scientific field — its own layer, toggleable
                independently, drawn under the coastline mask below so land
                is never colored as if it were an ocean grid cell. */}
            <TemperatureLayer
              grid={layer.grid}
              values={layer.values}
              colorMapper={colorMapper}
              visible={layerVisibility.temperatureField}
            />

            {/* Real 0.25° NEER scientific grid — drawn over the field so
                cell boundaries stay legible, kept subtle enough (low,
                fixed opacity) that it never competes with the field for
                attention; see GridOverlay.js for the alignment/zoom logic. */}
            <GridOverlay
              bounds={visibleBounds}
              scale={transform.scale}
              visible={layerVisibility.scientificGrid}
            />

            {graticuleLines}
            {coastlineElements}
            {domainBoundaryElement}
          </g>

          {/* Fixed-size overlay — graticule labels + hover/selection markers,
              positioned from world coordinates but not scaled with zoom. */}
          <g className="pointer-events-none">
            {layerVisibility.graticule &&
              latLines.map((lat) => {
                const { x, y } = worldToScreen(OCEAN_DOMAIN.lonMin, lat);
                if (x < -20 || x > VIEW_BOX.width + 20) return null;
                return (
                  <text
                    key={`lat-label-${lat}`}
                    x={Math.max(x, 4) + 3}
                    y={y}
                    dy="0.32em"
                    fontSize="9"
                    fontFamily="var(--font-mono)"
                    fill="rgba(174,197,214,0.75)"
                  >
                    {lat}°N
                  </text>
                );
              })}
            {layerVisibility.graticule &&
              lonLines.map((lon) => {
                const { x, y } = worldToScreen(lon, OCEAN_DOMAIN.latMin);
                if (y < -20 || y > VIEW_BOX.height + 20) return null;
                return (
                  <text
                    key={`lon-label-${lon}`}
                    x={x}
                    y={Math.min(y, VIEW_BOX.height - 4) - 4}
                    textAnchor="middle"
                    fontSize="9"
                    fontFamily="var(--font-mono)"
                    fill="rgba(174,197,214,0.75)"
                  >
                    {lon}°E
                  </text>
                );
              })}

            {hover && hoverInDomain && (
              <g transform={`translate(${worldToScreen(hover.lon, hover.lat).x} ${worldToScreen(hover.lon, hover.lat).y})`}>
                <line x1={-9} y1={0} x2={9} y2={0} stroke="#67E8F9" strokeWidth={1} opacity={0.85} />
                <line x1={0} y1={-9} x2={0} y2={9} stroke="#67E8F9" strokeWidth={1} opacity={0.85} />
                <circle r={3} fill="none" stroke="#67E8F9" strokeWidth={1.25} opacity={0.9} />
              </g>
            )}

            {activeSelected && (
              <g
                transform={`translate(${worldToScreen(activeSelected.lon, activeSelected.lat).x} ${
                  worldToScreen(activeSelected.lon, activeSelected.lat).y
                })`}
              >
                <circle r={9} fill="rgba(45,212,191,0.16)" stroke="#2DD4BF" strokeWidth={1.5} />
                <circle r={2.5} fill="#2DD4BF" />
              </g>
            )}
          </g>
        </svg>

        {/* Layer controls — temperature/anomaly/grid visibility + reset,
            grouped behind one compact trigger (MapControls) so the map
            surface itself stays uncluttered. */}
        <div className="absolute left-3 top-3">
          <MapControls layers={mapControlLayers} baseLayers={mapBaseLayers} onReset={resetView} />
        </div>

        {/* Zoom & domain reset controls */}
        <div className="absolute right-3 top-3 flex flex-col gap-1">
          <ToolbarButton icon={ZoomIn} label="Zoom in" onClick={() => zoomByButton(ZOOM_STEP)} />
          <ToolbarButton icon={ZoomOut} label="Zoom out" onClick={() => zoomByButton(1 / ZOOM_STEP)} />
          <ToolbarButton icon={RotateCcw} label="Reset / fit to NEER domain" onClick={resetView} />
        </div>

        {/* Scientific field — loading state. Non-blocking: the base map
            stays interactive while GET /reconstruct/grid is in flight. */}
        {layer.supported && layerVisibility.temperatureField && layer.isLoading && (
          <div className="pointer-events-none absolute left-1/2 top-3 flex -translate-x-1/2 items-center gap-2 rounded-full border border-border-strong bg-surface-overlay/95 px-3 py-1.5 shadow-raised">
            <LoadingSkeleton
              variant="text"
              lines={1}
              label={`Loading ${activeLabel?.toLowerCase()} field`}
              className="w-40"
            />
          </div>
        )}

        {/* Scientific field — error state, with a retry that re-issues the
            same GET /reconstruct/grid request. */}
        {layer.supported && layerVisibility.temperatureField && layer.isError && (
          <div className="absolute inset-x-4 top-3 z-raised flex justify-center">
            <ErrorState
              className="w-full max-w-sm bg-surface-overlay/95 py-5 shadow-raised"
              title="Field data unavailable"
              message={layer.error?.message ?? "The scientific field could not be loaded."}
              onRetry={layer.retry}
            />
          </div>
        )}

        {/* Scientific field — missing-data state: the request succeeded
            but this date/depth genuinely has nothing to draw (e.g. no
            climatology loaded, so anomaly can't be derived) — never
            silently replaced with a fabricated value. */}
        {layerVisibility.temperatureField && showEmptyFieldState && (
          <div className="pointer-events-none absolute left-1/2 top-3 -translate-x-1/2 rounded-full border border-border-strong bg-surface-overlay/95 px-3 py-1.5 shadow-raised">
            <StatusIndicator
              status="warning"
              label={`No ${activeLabel?.toLowerCase()} field available for this date`}
              size="sm"
            />
          </div>
        )}

        {/* Compact scientific hover tooltip (Phase 35C-B, Requirement 1).
            Reuses the same visual tokens as the shared Tooltip component
            (border-border-strong/bg-surface-overlay/shadow-raised/
            text-caption) rather than wrapping Tooltip itself: Tooltip's
            hover-a-fixed-child, fixed-position API doesn't fit a value that
            already tracks a moving SVG cursor across arbitrary lat/lon —
            same reasoning as the loading/error pills just above, which are
            also inline markup in that same visual language rather than a
            second Tooltip instance. Positioned from `hoverScreenPos`
            (container-relative pixels), flipped to whichever side of the
            cursor keeps it inside the map surface, so it never obstructs
            the point currently being read (Requirement 1's "avoid
            obstructing important map content"). Land/outside-domain never
            show a temperature/anomaly number — only `describePointStatus`'s
            wording — so this can never look like a fabricated ocean
            reading (Requirement 5 / scientific-integrity). */}
        {hover && hoverScreenPos && (
          <div
            className="pointer-events-none absolute z-toast flex w-56 flex-col gap-1.5 rounded-md border border-border-strong bg-surface-overlay px-3 py-2 text-caption text-text-primary shadow-raised"
            style={{
              left: hoverScreenPos.x,
              top: hoverScreenPos.y,
              transform: `translate(${
                containerRef.current && hoverScreenPos.x > containerRef.current.clientWidth / 2
                  ? "calc(-100% - 14px)"
                  : "14px"
              }, ${
                containerRef.current && hoverScreenPos.y > containerRef.current.clientHeight / 2
                  ? "calc(-100% - 14px)"
                  : "14px"
              })`,
            }}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-mono text-small font-semibold text-text-primary">
                {formatLat(hover.lat)}, {formatLon(hover.lon)}
              </span>
              <StatusIndicator
                status={
                  hoverStatus === POINT_STATUS.OCEAN
                    ? "online"
                    : hoverStatus === POINT_STATUS.LAND
                      ? "offline"
                      : "warning"
                }
                showLabel={false}
                size="sm"
              />
            </div>

            {hoverStatus !== POINT_STATUS.OCEAN ? (
              <p className="text-text-muted">{describePointStatus(hoverStatus)}</p>
            ) : (
              <>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-text-muted">Temperature</span>
                  <span className="font-mono text-text-primary">
                    {hoverTemperature === null ? "-- (no data)" : `${hoverTemperature.toFixed(2)}°C`}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-text-muted">Anomaly</span>
                  <span className="font-mono text-text-primary">
                    {hoverAnomaly === null ? "-- (no data)" : `${formatSigned(hoverAnomaly)}°C`}
                  </span>
                </div>
              </>
            )}

            <div className="flex items-center justify-between gap-3 border-t border-border-subtle pt-1 text-text-muted">
              <span className="flex items-center gap-1">
                <CalendarDays size={10} strokeWidth={1.75} aria-hidden="true" />
                {activeDate ? activeDate.toLocaleDateString("en-GB", DATE_FORMAT) : "No date selected"}
              </span>
              <span className="flex items-center gap-1">
                <MoveVertical size={10} strokeWidth={1.75} aria-hidden="true" />
                {formatDepth(depth)}
              </span>
            </div>
          </div>
        )}

        {/* Legend */}
        <div className="absolute bottom-3 left-3 flex flex-col gap-2 rounded-md border border-border bg-surface-base/90 px-3 py-2 backdrop-blur-sm max-w-[calc(100%-1.5rem)] sm:max-w-xs md:max-w-sm">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-caption text-text-muted">
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: "#0E2A42" }} aria-hidden="true" />
              Ocean (data domain)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: "#132133" }} aria-hidden="true" />
              Land (masked)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full border border-accent-400" aria-hidden="true" />
              Selected cell
            </span>
          </div>

          <MapLegend
            variable={activeVariable}
            activeLabel={activeLabel}
            colorConfig={colorConfig}
            sample={legendSample}
            extent={fieldExtent}
            layer={layer}
            fieldVisible={layerVisibility.temperatureField}
          />
        </div>
      </div>

      {/* Coordinate readout */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-1 border-t border-border-subtle pt-3 text-caption font-mono text-text-muted">
        <CoordinateDisplay lat={readoutPoint?.lat} lon={readoutPoint?.lon} />
        <span>DEPTH {depth} m</span>
        <span>VALUE {readoutValueText}</span>
        {(hover || activeSelected) && readoutStatus && (
          <Badge
            variant={
              readoutStatus === POINT_STATUS.OCEAN
                ? "accent"
                : readoutStatus === POINT_STATUS.OUTSIDE_DOMAIN
                  ? "warning"
                  : "neutral"
            }
            size="sm"
          >
            {readoutStatus === POINT_STATUS.OCEAN ? (
              "Ocean grid cell"
            ) : (
              <span className="flex items-center gap-1">
                <Compass size={10} strokeWidth={2} aria-hidden="true" /> {describePointStatus(readoutStatus)}
              </span>
            )}
          </Badge>
        )}
      </div>
    </Panel>
  );
}