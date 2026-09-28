// -----------------------------------------------------------------------------
// PHASE 35D1 — MapLegend / MapControls / LayerToggle / CoordinateDisplay tests
//
// Two layers: pure-logic tests for the scientific-integrity rules (what range
// the legend may show, and that its labels match the colors the map paints),
// and server-render tests that actually mount each component through every
// state — a component that throws on a branch (e.g. an undefined variable in
// an empty-state) is invisible to helper-only tests.
// -----------------------------------------------------------------------------
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Thermometer } from "lucide-react";
import { divergingColor, getDivergingLimit, makeColorMapper, thermalColor } from "@/lib/colorScale";
import CoordinateDisplay, { formatCoordWithPrecision } from "./CoordinateDisplay";
import { GRID_MIN_VISIBLE_SCALE } from "./GridOverlay";
import LayerToggle from "./LayerToggle";
import MapControls from "./MapControls";
import MapLegend, { LEGEND_STATE, deriveLegendState } from "./MapLegend";
import TemperatureLegend, { legendDigits } from "./TemperatureLegend";
import { subtractGrids } from "./useOceanMapLayer";

const html = (el) => renderToStaticMarkup(el);
// Strip tags so assertions read the text a user would see.
const text = (el) => html(el).replace(/<[^>]*>/g, " ").replace(/\s+/g, " ");

const SST_LAYER = {
  supported: true,
  status: "success",
  isLoading: false,
  isError: false,
  values: [
    [24.5, 26.1, null],
    [27.9, null, 30.2],
  ],
  grid: { lat: [10, 10.25], lon: [70, 70.25, 70.5] },
};

describe("deriveLegendState — never shows a scale without real data", () => {
  const base = { fieldVisible: true, isAnomaly: false, layer: null, extent: null };
  const EXT = { min: 24.5, max: 30.2 };

  it("READY only when a real extent exists", () => {
    expect(deriveLegendState({ ...base, extent: EXT }).state).toBe(LEGEND_STATE.READY);
    expect(deriveLegendState({ ...base, layer: { status: "success" } }).state).toBe(LEGEND_STATE.EMPTY);
  });

  it("HIDDEN wins over everything when the layer is toggled off", () => {
    const s = deriveLegendState({ ...base, fieldVisible: false, extent: EXT, isError: true });
    expect(s.state).toBe(LEGEND_STATE.HIDDEN);
  });

  it("UNSUPPORTED layer (no backend grid) shows its reason, never a scale", () => {
    const s = deriveLegendState({
      ...base,
      layer: { supported: false, reason: "No backend grid endpoint yet for salinity" },
    });
    expect(s.state).toBe(LEGEND_STATE.UNSUPPORTED);
    expect(s.message).toContain("salinity");
  });

  it("ERROR takes precedence over a stale extent", () => {
    const s = deriveLegendState({ ...base, extent: EXT, isError: true, error: new Error("boom") });
    expect(s.state).toBe(LEGEND_STATE.ERROR);
    expect(s.message).toBe("boom");
  });

  it("LOADING with no data yet -> loading skeleton; with stale data -> READY but flagged updating", () => {
    expect(deriveLegendState({ ...base, isLoading: true }).state).toBe(LEGEND_STATE.LOADING);
    const stale = deriveLegendState({ ...base, extent: EXT, isLoading: true });
    expect(stale.state).toBe(LEGEND_STATE.READY);
    expect(stale.updating).toBe(true);
  });

  it("distinguishes region-too-large, no-date and missing-climatology", () => {
    const zoom = deriveLegendState({
      ...base,
      layer: { tooLargeForCellLimit: true, status: "idle", estimatedCells: 24341 },
    });
    expect(zoom.state).toBe(LEGEND_STATE.ZOOM);
    // Full NEER domain: 241 x 101 cells vs the backend's 4096-cell limit.
    expect(zoom.message).toContain("24,341");
    expect(zoom.message).toContain("4,096");
    expect(deriveLegendState({ ...base, layer: { tooLargeForCellLimit: true, status: "idle" } }).state).toBe(
      LEGEND_STATE.ZOOM
    );
    expect(deriveLegendState({ ...base, layer: { status: "idle" } }).state).toBe(LEGEND_STATE.NO_DATE);
    const anomaly = deriveLegendState({ ...base, isAnomaly: true, layer: { status: "success", values: null } });
    expect(anomaly.state).toBe(LEGEND_STATE.EMPTY);
    expect(anomaly.message).toMatch(/climatology/i);
  });
});

describe("legend labels match the colors the map actually paints", () => {
  it("diverging limit is symmetric max(|min|,|max|) and shared with makeColorMapper", () => {
    const extent = { min: -0.5, max: 2.0 };
    expect(getDivergingLimit(extent)).toBe(2.0);
    expect(getDivergingLimit({ min: -3, max: 1 })).toBe(3);
    expect(getDivergingLimit({ min: 0, max: 0 })).toBe(1); // degenerate: avoid /0

    // The value at the legend's "+L" end is painted with the scale's max color,
    // and the value at "-L" with its min color — i.e. labels == paint.
    const paint = makeColorMapper("diverging", extent);
    expect(paint(2.0)).toBe(divergingColor(1));
    expect(paint(-2.0)).toBe(divergingColor(-1));
    expect(paint(0)).toBe(divergingColor(0));
    // Asymmetric data: the real minimum (-0.5) is NOT the scale's deep-blue end.
    expect(paint(-0.5)).not.toBe(divergingColor(-1));
  });

  it("sequential legend spans exactly [min, max] of the loaded data", () => {
    const paint = makeColorMapper("sequential", { min: 24.5, max: 30.2 });
    expect(paint(24.5)).toBe(thermalColor(0));
    expect(paint(30.2)).toBe(thermalColor(1));
  });

  it("TemperatureLegend labels diverging ends as -L / 0 / +L, and reports the true data span", () => {
    const out = text(<TemperatureLegend kind="diverging" extent={{ min: -0.5, max: 2.0 }} unit="°C" />);
    expect(out).toContain("-2.0 °C");
    expect(out).toContain("0.0 °C");
    expect(out).toContain("+2.0 °C");
    expect(out).toContain("Loaded data: -0.5 to +2.0 °C");
  });

  it("TemperatureLegend labels sequential ends with the real min/max and units", () => {
    const out = text(<TemperatureLegend kind="sequential" extent={{ min: 24.5, max: 30.2 }} unit="°C" />);
    expect(out).toContain("24.5 °C");
    expect(out).toContain("30.2 °C");
    expect(out).toContain("27.4 °C"); // midpoint of the real range
  });

  it("renders nothing (no invented range) when the extent is missing or non-finite", () => {
    expect(html(<TemperatureLegend extent={null} />)).toBe("");
    expect(html(<TemperatureLegend extent={{ min: NaN, max: 3 }} />)).toBe("");
  });

  it("uses more decimals for narrow ranges so they don't collapse to 0.0", () => {
    expect(legendDigits(10)).toBe(1);
    expect(legendDigits(0.5)).toBe(2);
    expect(legendDigits(0.06)).toBe(3);
    const out = text(<TemperatureLegend kind="diverging" extent={{ min: -0.03, max: 0.04 }} />);
    expect(out).toContain("+0.040");
  });
});

describe("MapLegend — renders every state without inventing values", () => {
  it("READY (SST): real range, unit, cell count", () => {
    const out = text(<MapLegend variable="sst" activeLabel="Sea Surface Temp." layer={SST_LAYER} />);
    expect(out).toContain("Sea Surface Temp.");
    expect(out).toContain("24.5 °C");
    expect(out).toContain("30.2 °C");
    expect(out).toContain("2×3 cells");
  });

  it("READY (anomaly): diverging scale with cooler/warmer sides", () => {
    const anomaly = { ...SST_LAYER, values: [[-0.4, 0.1, null], [1.2, null, -0.9]] };
    const out = text(<MapLegend variable="anomaly" activeLabel="Anomaly" layer={anomaly} />);
    expect(out).toContain("cooler");
    expect(out).toContain("warmer");
    expect(out).toContain("-1.2 °C");
    expect(out).toContain("+1.2 °C");
  });

  it("switching the active layer switches the legend (no leftover SST scale)", () => {
    const sst = text(<MapLegend variable="sst" activeLabel="Sea Surface Temp." layer={SST_LAYER} />);
    const anomaly = text(<MapLegend variable="anomaly" activeLabel="Anomaly" layer={{ ...SST_LAYER, values: [[-1, 1]] }} />);
    expect(sst).not.toContain("cooler");
    expect(anomaly).toContain("cooler");
  });

  it("LOADING / ERROR / EMPTY / HIDDEN / UNSUPPORTED show a message and no numbers", () => {
    const cases = [
      [{ layer: { supported: true, isLoading: true, status: "loading", values: null } }, /Loading/],
      [{ layer: { supported: true, isError: true, status: "error", error: { message: "Backend down" } } }, /Backend down/],
      [{ layer: { supported: true, status: "success", values: null } }, /No data in the loaded view/],
      [{ layer: SST_LAYER, fieldVisible: false }, /Layer hidden/],
      [{ variable: "sss", activeLabel: "Salinity", layer: { supported: false, reason: "No backend grid endpoint yet for salinity" } }, /salinity/],
    ];
    for (const [props, pattern] of cases) {
      const out = text(<MapLegend variable="sst" activeLabel="Sea Surface Temp." {...props} />);
      expect(out).toMatch(pattern);
      expect(out).not.toMatch(/\d+\.\d\s*°C/); // no scale numbers
      expect(html(<MapLegend variable="sst" {...props} />)).not.toContain("linearGradient");
    }
  });

  it("an unsupported variable never borrows the SST color scale", () => {
    const out = html(<MapLegend variable="sss" activeLabel="Salinity" layer={{ supported: false, reason: "n/a" }} />);
    expect(out).not.toContain("linearGradient");
    expect(out).not.toContain("°C");
  });

  it("renders children (map key) and exposes a labelled, collapsible region", () => {
    const out = html(
      <MapLegend variable="sst" layer={SST_LAYER}>
        <span>Ocean key</span>
      </MapLegend>
    );
    expect(out).toContain("Ocean key");
    expect(out).toContain('aria-label="Map legend"');
    expect(out).toContain('aria-expanded="true"');
  });
});

describe("CoordinateDisplay", () => {
  it("shows real coordinates with 2-decimal cardinal formatting", () => {
    const out = text(<CoordinateDisplay lat={15.25} lon={73.5} />);
    expect(out).toContain("15.25°N");
    expect(out).toContain("73.50°E");
  });

  it("shows placeholders for missing input rather than 0", () => {
    const out = text(<CoordinateDisplay lat={null} lon={undefined} />);
    expect(out.match(/--\.--°/g)).toHaveLength(2);
    expect(out).not.toContain("0.00");
  });

  it("supports precision, decimal format, point alias and a source label", () => {
    expect(text(<CoordinateDisplay point={{ lat: 12.3456, lon: 80.1234 }} precision={3} />)).toContain("12.346°N");
    expect(text(<CoordinateDisplay lat={-2.5} lon={80} format="decimal" />)).toContain("−2.50°");
    expect(text(<CoordinateDisplay lat={1} lon={2} source="Cursor" />)).toContain("Cursor");
  });

  it("is not a live region (it changes on every pointer move)", () => {
    const out = html(<CoordinateDisplay lat={1} lon={2} />);
    expect(out).not.toContain('role="status"');
    expect(out).not.toContain("aria-live");
  });

  it("formatCoordWithPrecision matches formatLat/Lon at the default precision", () => {
    expect(formatCoordWithPrecision(-2.5, "N", "S")).toBe("2.50°S");
  });
});

describe("LayerToggle", () => {
  it("icon layout exposes state via aria-pressed and an accessible name", () => {
    const on = html(<LayerToggle icon={Thermometer} label="Temperature layer" active onToggle={() => {}} />);
    const off = html(<LayerToggle icon={Thermometer} label="Temperature layer" onToggle={() => {}} />);
    expect(on).toContain('aria-pressed="true"');
    expect(off).toContain('aria-pressed="false"');
    expect(on).toContain('aria-label="Temperature layer"');
  });

  it("row layout is a switch with visible description, linked via aria-describedby", () => {
    const out = html(
      <LayerToggle layout="row" icon={Thermometer} label="Temperature layer" description="Sea-surface temperature field" active onToggle={() => {}} />
    );
    expect(out).toContain('role="switch"');
    expect(out).toContain('aria-checked="true"');
    expect(out).toContain("Sea-surface temperature field");
    expect(out).toContain("aria-describedby");
  });

  it("disabled row is really disabled and shows the reason instead of the description", () => {
    const out = html(
      <LayerToggle layout="row" label="X" description="desc" disabled disabledReason="No data yet" onToggle={() => {}} />
    );
    expect(out).toContain("disabled");
    expect(out).toContain("No data yet");
    expect(out).not.toContain(">desc<");
  });
});

describe("MapControls", () => {
  it("renders a closed disclosure trigger with correct ARIA and live layer count", () => {
    const out = html(
      <MapControls
        temperatureVisible
        anomalyVisible={false}
        gridVisible
        onToggleTemperature={() => {}}
        onToggleAnomaly={() => {}}
        onToggleGrid={() => {}}
        onReset={() => {}}
      />
    );
    expect(out).toContain('aria-label="Map layers"');
    expect(out).toContain('aria-expanded="false"');
    expect(out).toContain("aria-controls");
    expect(out).not.toContain("aria-pressed"); // disclosure, not a toggle
    expect(out).not.toContain('role="menu"');
  });

  it("grid overlay only draws above a known zoom; the threshold is exported for the UI to explain it", () => {
    expect(GRID_MIN_VISIBLE_SCALE).toBeGreaterThan(1);
  });
});

describe("subtractGrids — anomaly derivation never fabricates values", () => {
  it("subtracts matching cells", () => {
    expect(subtractGrids([[28, 29]], [[27, 27.5]])).toEqual([[1, 1.5]]);
  });

  it("a missing climatology cell yields null, NOT the raw temperature", () => {
    expect(subtractGrids([[28, 29]], [[27, null]])).toEqual([[1, null]]);
  });

  it("a masked temperature cell yields null and does not throw", () => {
    expect(subtractGrids([[null, 29]], [[27, 27]])).toEqual([[null, 2]]);
    expect(subtractGrids([[NaN]], [[1]])).toEqual([[null]]);
  });

  it("handles 3D (multi-depth) grids and a shorter climatology safely", () => {
    expect(subtractGrids([[[20, 19]]], [[[18, 18]]])).toEqual([[[2, 1]]]);
    expect(subtractGrids([[1], [2]], [[1]])).toEqual([[0], [null]]);
  });
});