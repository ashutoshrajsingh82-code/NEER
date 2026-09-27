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
// / visible viewport, still with no rendering wired up (the spec repeats
// Phase 34C's "do not implement the temperature or anomaly data layers
// yet") — see that file's header for the variable -> backend-support table
// and the cell-limit check. Its result only powers the legend's status
// line below; a future phase turns `layer.values` into an actual overlay.
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
  Maximize2,
  Minimize2,
  MoveVertical,
  Navigation,
  RotateCcw,
  Thermometer,
  TrendingUp,
  Waves,
  Wind,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { Badge, Button, Panel, Tabs, Tooltip } from "@/components/ui";
import { cn } from "@/lib/cn";
import {
  OCEAN_DOMAIN,
  clamp,
  formatDepth,
  formatLat,
  formatLon,
  round2,
  snapToGrid,
} from "@/lib/oceanDomain";
import { LANDMASSES, VIEW_BOX, isOnLand, project, unproject } from "./landmask";
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
const FINE_GRID_MIN_SCALE = 3;
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
 * Turns a useOceanMapLayer() result into the legend's one-line status —
 * this is the only place its result is read; nothing here renders the
 * fetched grid itself (see useOceanMapLayer.js's header for why).
 */
function describeLayerStatus(layer, activeLabel) {
  if (!layer.supported) {
    return `${activeLabel} · ${layer.reason} — demo interaction only`;
  }
  if (layer.tooLargeForCellLimit) {
    return `${activeLabel} · Zoom in to load live grid data (visible region too large for one request)`;
  }
  if (layer.isLoading) {
    return `${activeLabel} · Loading live grid data…`;
  }
  if (layer.isError) {
    return `${activeLabel} · Grid request failed — ${layer.error?.message ?? "unknown error"}`;
  }
  if (layer.values) {
    const rows = layer.grid?.lat?.length ?? 0;
    const cols = layer.grid?.lon?.length ?? 0;
    return `${activeLabel} · Live grid loaded (${rows}×${cols} cells) — visualization not yet implemented`;
  }
  if (layer.status === "success") {
    return `${activeLabel} · No climatology for this date — anomaly unavailable`;
  }
  return `${activeLabel} · No date selected — demo interaction only`;
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
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [controlsOpen, setControlsOpen] = useState(false);
  const [layerVisibility, setLayerVisibility] = useState({
    graticule: true,
    fineGrid: true,
    coastline: true,
    domainBoundary: true,
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
    selectPoint(snapToGrid(lat, lon));
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

  // Close the compact layer-controls popover on outside click.
  const controlsRef = useRef(null);
  useEffect(() => {
    if (!controlsOpen) return undefined;
    function handleOutside(event) {
      if (controlsRef.current && !controlsRef.current.contains(event.target)) {
        setControlsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleOutside);
    return () => document.removeEventListener("mousedown", handleOutside);
  }, [controlsOpen]);

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

  const hoverOnLand = hover ? isOnLand(hover.lat, hover.lon) : false;
  const selectedOnLand = activeSelected ? isOnLand(activeSelected.lat, activeSelected.lon) : false;
  const hoverInDomain =
    hover &&
    hover.lat >= OCEAN_DOMAIN.latMin - 2 &&
    hover.lat <= OCEAN_DOMAIN.latMax + 2 &&
    hover.lon >= OCEAN_DOMAIN.lonMin - 2 &&
    hover.lon <= OCEAN_DOMAIN.lonMax + 2;

  const readoutLat = hover ? formatLat(hover.lat) : activeSelected ? formatLat(activeSelected.lat) : "--.--°";
  const readoutLon = hover ? formatLon(hover.lon) : activeSelected ? formatLon(activeSelected.lon) : "--.--°";

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
            icon={Layers}
            label="Map layer controls"
            active={controlsOpen}
            onClick={() => setControlsOpen((prev) => !prev)}
          />
          <ToolbarButton icon={RotateCcw} label="Reset view" onClick={resetView} />
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
        onPointerLeave={() => setHover(null)}
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
            <pattern id="fineGrid" width="4" height="4" patternUnits="userSpaceOnUse">
              <path d="M 4 0 L 0 0 0 4" fill="none" stroke="rgba(148,197,224,0.16)" strokeWidth="0.15" />
            </pattern>
          </defs>

          {/* Pannable/zoomable map content — world-space coordinates */}
          <g transform={`translate(${transform.x} ${transform.y}) scale(${transform.scale})`}>
            <rect x={0} y={0} width={VIEW_BOX.width} height={VIEW_BOX.height} fill="url(#oceanGradient)" />

            {layerVisibility.fineGrid && transform.scale >= FINE_GRID_MIN_SCALE && (
              <rect x={0} y={0} width={VIEW_BOX.width} height={VIEW_BOX.height} fill="url(#fineGrid)" />
            )}

            {layerVisibility.graticule &&
              latLines.map((lat) => {
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
            {layerVisibility.graticule &&
              lonLines.map((lon) => {
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

            {layerVisibility.coastline &&
              LANDMASSES.map((mass) => (
                <polygon
                  key={mass.id}
                  points={mass.points.map((p) => project(p).join(",")).join(" ")}
                  fill="#132133"
                  stroke="rgba(148,197,224,0.3)"
                  strokeWidth={0.75 / transform.scale}
                />
              ))}

            {layerVisibility.domainBoundary && (
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
            )}
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

        {/* Compact layer-controls popover */}
        {controlsOpen && (
          <div
            ref={controlsRef}
            className="absolute left-3 top-3 z-raised flex w-48 flex-col gap-1 rounded-md border border-border-strong bg-surface-overlay p-2.5 shadow-raised"
          >
            <p className="px-1 pb-1 text-caption uppercase tracking-widest text-text-muted">Map layers</p>
            {[
              { key: "graticule", label: "Graticule & labels" },
              { key: "fineGrid", label: "0.25° grid (on zoom)" },
              { key: "coastline", label: "Coastline" },
              { key: "domainBoundary", label: "Domain boundary" },
            ].map((item) => (
              <label
                key={item.key}
                className="flex cursor-pointer items-center justify-between gap-2 rounded-sm px-1.5 py-1 text-small text-text-secondary hover:bg-surface-raised"
              >
                {item.label}
                <input
                  type="checkbox"
                  checked={layerVisibility[item.key]}
                  onChange={() => toggleLayer(item.key)}
                  className="h-3.5 w-3.5 accent-accent-400"
                />
              </label>
            ))}
            <p className="mt-1 border-t border-border-subtle px-1 pt-1.5 text-caption text-text-disabled">
              Reserved for future layers: SST, SSS, SSH/SLA, currents, winds, subsurface temp., anomaly,
              uncertainty.
            </p>
          </div>
        )}

        {/* Zoom controls */}
        <div className="absolute right-3 top-3 flex flex-col gap-1">
          <ToolbarButton icon={ZoomIn} label="Zoom in" onClick={() => zoomByButton(ZOOM_STEP)} />
          <ToolbarButton icon={ZoomOut} label="Zoom out" onClick={() => zoomByButton(1 / ZOOM_STEP)} />
        </div>

        {/* Legend */}
        <div className="absolute bottom-3 left-3 flex flex-col gap-1.5 rounded-md border border-border bg-surface-base/90 px-3 py-2 backdrop-blur-sm">
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
          <p className="text-caption text-text-disabled">{describeLayerStatus(layer, activeLabel)}</p>
        </div>
      </div>

      {/* Coordinate readout */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-1 border-t border-border-subtle pt-3 text-caption font-mono text-text-muted">
        <span>LAT {readoutLat}</span>
        <span>LON {readoutLon}</span>
        <span>DEPTH {depth} m</span>
        <span>VALUE --</span>
        {(hover || activeSelected) && (
          <Badge variant={(hover ? hoverOnLand : selectedOnLand) ? "neutral" : "accent"} size="sm">
            {(hover ? hoverOnLand : selectedOnLand) ? (
              <span className="flex items-center gap-1">
                <Compass size={10} strokeWidth={2} aria-hidden="true" /> Land — outside reconstruction domain
              </span>
            ) : (
              "Ocean grid cell"
            )}
          </Badge>
        )}
      </div>
    </Panel>
  );
}