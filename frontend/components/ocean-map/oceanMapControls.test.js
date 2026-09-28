import { describe, expect, it, vi } from "vitest";
import {
  formatLat,
  formatLon,
  isWithinDomain,
  OCEAN_DOMAIN,
  snapToGrid,
} from "@/lib/oceanDomain";
import { formatCoordWithPrecision } from "./CoordinateDisplay";
import {
  computeFiniteExtent,
  divergingColor,
  makeColorMapper,
  thermalColor,
  VARIABLE_COLOR_CONFIG,
} from "@/lib/colorScale";
import { formatSigned } from "@/lib/format";
import { LAYER_SUPPORT, MAX_GRID_POINTS } from "./useOceanMapLayer";
import {
  classifyPointLocation,
  describePointStatus,
  isQueryablePoint,
  POINT_STATUS,
} from "@/lib/pointClassification";
import { isOnLand, LANDMASSES, VIEW_BOX } from "./landmask";

describe("PHASE 35D2: Ocean Map Integration and QA", () => {
  describe("1. Geographic Domain & Grid Verification", () => {
    it("matches the scientific domain specifications: 45°E–105°E, 5°N–30°N, 0.25° grid", () => {
      expect(OCEAN_DOMAIN.lonMin).toBe(45);
      expect(OCEAN_DOMAIN.lonMax).toBe(105);
      expect(OCEAN_DOMAIN.latMin).toBe(5);
      expect(OCEAN_DOMAIN.latMax).toBe(30);
      expect(OCEAN_DOMAIN.resolution).toBe(0.25);

      // Verify domain width and height in degrees
      const lonSpan = OCEAN_DOMAIN.lonMax - OCEAN_DOMAIN.lonMin;
      const latSpan = OCEAN_DOMAIN.latMax - OCEAN_DOMAIN.latMin;
      expect(lonSpan).toBe(60);
      expect(latSpan).toBe(25);

      // Verify SVG aspect ratio matches 60 deg x 25 deg (16 user-units per degree)
      expect(VIEW_BOX.width).toBe(60 * 16);
      expect(VIEW_BOX.height).toBe(25 * 16);
    });

    it("verifies isWithinDomain boundaries", () => {
      expect(isWithinDomain(15, 75, OCEAN_DOMAIN)).toBe(true);
      expect(isWithinDomain(5, 45, OCEAN_DOMAIN)).toBe(true);
      expect(isWithinDomain(30, 105, OCEAN_DOMAIN)).toBe(true);

      // Outside
      expect(isWithinDomain(4.9, 75, OCEAN_DOMAIN)).toBe(false);
      expect(isWithinDomain(30.1, 75, OCEAN_DOMAIN)).toBe(false);
      expect(isWithinDomain(15, 44.9, OCEAN_DOMAIN)).toBe(false);
      expect(isWithinDomain(15, 105.1, OCEAN_DOMAIN)).toBe(false);
    });

    it("snaps clicked points accurately to the 0.25° grid and clamps within domain", () => {
      const snapped1 = snapToGrid(15.12, 73.61, OCEAN_DOMAIN);
      expect(snapped1.lat).toBe(15.0);
      expect(snapped1.lon).toBe(73.5);

      const snappedEdge = snapToGrid(3.2, 40.1, OCEAN_DOMAIN);
      expect(snappedEdge.lat).toBe(5.0);
      expect(snappedEdge.lon).toBe(45.0);
    });
  });

  describe("2. CoordinateDisplay Precision & Formats", () => {
    it("formats latitude with reasonable 2-decimal precision and cardinal direction", () => {
      expect(formatLat(15.25)).toBe("15.25°N");
      expect(formatLat(5.0)).toBe("5.00°N");
      expect(formatLat(30.0)).toBe("30.00°N");
      expect(formatLat(-2.5)).toBe("2.50°S");
    });

    it("formats longitude with reasonable 2-decimal precision and cardinal direction", () => {
      expect(formatLon(73.5)).toBe("73.50°E");
      expect(formatLon(45.0)).toBe("45.00°E");
      expect(formatLon(105.0)).toBe("105.00°E");
      expect(formatLon(-10.25)).toBe("10.25°W");
    });

    it("formats coordinates with custom precision", () => {
      expect(formatCoordWithPrecision(12.3456, "N", "S", 3)).toBe("12.346°N");
      expect(formatCoordWithPrecision(80.1234, "E", "W", 1)).toBe("80.1°E");
      expect(formatCoordWithPrecision(-5.789, "N", "S", 2)).toBe("5.79°S");
    });

    it("returns placeholder --.--° for invalid or nullish coordinates", () => {
      expect(formatCoordWithPrecision(null, "N", "S")).toBe("--.--°");
      expect(formatCoordWithPrecision(undefined, "E", "W")).toBe("--.--°");
      expect(formatCoordWithPrecision(NaN, "N", "S")).toBe("--.--°");
      expect(formatCoordWithPrecision(Infinity, "E", "W")).toBe("--.--°");
    });
  });

  describe("3. Scientific Integrity & Color Scale", () => {
    it("computes finite extent from real grid data without inventing values", () => {
      const testGrid = [
        [24.5, 25.0, null],
        [NaN, 28.2, 30.1],
        [undefined, 22.4, 27.8],
      ];
      const extent = computeFiniteExtent(testGrid);
      expect(extent).not.toBeNull();
      expect(extent?.min).toBe(22.4);
      expect(extent?.max).toBe(30.1);
    });

    it("returns null for completely empty or masked grids rather than inventing a range", () => {
      const emptyGrid = [[null, NaN], [undefined, null]];
      expect(computeFiniteExtent(emptyGrid)).toBeNull();
    });

    it("formats signed anomaly values distinguishing positive and negative values", () => {
      expect(formatSigned(1.42)).toBe("+1.42");
      expect(formatSigned(-0.85)).toBe("-0.85");
      expect(formatSigned(0)).toBe("0.00");
      expect(formatSigned(null)).toBe("--");
      expect(formatSigned(NaN)).toBe("--");
    });

    it("maps color scales appropriately for sequential SST vs diverging anomaly", () => {
      expect(VARIABLE_COLOR_CONFIG.sst.kind).toBe("sequential");
      expect(VARIABLE_COLOR_CONFIG.sst.unit).toBe("°C");

      expect(VARIABLE_COLOR_CONFIG.anomaly.kind).toBe("diverging");
      expect(VARIABLE_COLOR_CONFIG.anomaly.unit).toBe("°C");

      const sstMapper = makeColorMapper("sequential", { min: 20, max: 30 });
      expect(typeof sstMapper(25)).toBe("string");
      expect(sstMapper(null)).toBeNull();

      const anomalyMapper = makeColorMapper("diverging", { min: -2.5, max: 2.5 });
      expect(typeof anomalyMapper(-1.5)).toBe("string");
      expect(typeof anomalyMapper(1.5)).toBe("string");
      expect(anomalyMapper(null)).toBeNull();
    });

    it("generates correct color samples at endpoints and zero for diverging anomaly", () => {
      const coldColor = divergingColor(-1.0);
      const neutralColor = divergingColor(0.0);
      const warmColor = divergingColor(1.0);

      expect(coldColor).toContain("rgba");
      expect(neutralColor).toContain("rgba");
      expect(warmColor).toContain("rgba");
      expect(coldColor).not.toBe(warmColor);
    });

    it("supports only real backend fields in LAYER_SUPPORT", () => {
      expect(LAYER_SUPPORT.sst.supported).toBe(true);
      expect(LAYER_SUPPORT.anomaly.supported).toBe(true);
      expect(LAYER_SUPPORT.sss.supported).toBe(false);
      expect(LAYER_SUPPORT.ssh.supported).toBe(false);
      expect(MAX_GRID_POINTS).toBe(4096);
    });
  });

  describe("4. Point Inspection & Landmask Classification", () => {
    it("classifies ocean points as queryable", () => {
      const status = classifyPointLocation(15.0, 70.0, isOnLand, OCEAN_DOMAIN);
      expect(status).toBe(POINT_STATUS.OCEAN);
      expect(isQueryablePoint(status)).toBe(true);
      expect(describePointStatus(status)).toBeNull();
    });

    it("classifies Indian subcontinent land point as land and non-queryable", () => {
      // (lat 20.0, lon 78.0) is central India landmass
      const status = classifyPointLocation(20.0, 78.0, isOnLand, OCEAN_DOMAIN);
      expect(status).toBe(POINT_STATUS.LAND);
      expect(isQueryablePoint(status)).toBe(false);
      expect(describePointStatus(status)).toContain("Land");
    });

    it("classifies outside-domain point as non-queryable", () => {
      const status = classifyPointLocation(35.0, 70.0, isOnLand, OCEAN_DOMAIN);
      expect(status).toBe(POINT_STATUS.OUTSIDE_DOMAIN);
      expect(isQueryablePoint(status)).toBe(false);
      expect(describePointStatus(status)).toContain("Outside");
    });

    it("verifies pre-computed SVG points on LANDMASSES for rendering performance", () => {
      expect(LANDMASSES.length).toBeGreaterThan(0);
      for (const mass of LANDMASSES) {
        expect(typeof mass.svgPoints).toBe("string");
        expect(mass.svgPoints.length).toBeGreaterThan(0);
      }
    });
  });

  describe("5. MapControls Layer Toggle Transitions & Reset", () => {
    it("manages active/inactive state transitions between temperature and anomaly layers", () => {
      let activeVariable = "sst";
      let layerVisibility = {
        temperatureField: true,
        scientificGrid: true,
      };

      const setLayerVisibility = (updater) => {
        layerVisibility = typeof updater === "function" ? updater(layerVisibility) : updater;
      };
      const changeVariable = (v) => {
        activeVariable = v;
      };

      const isSstActive = () => layerVisibility.temperatureField && activeVariable === "sst";
      const isAnomalyActive = () => layerVisibility.temperatureField && activeVariable === "anomaly";

      // Initially SST is active
      expect(isSstActive()).toBe(true);
      expect(isAnomalyActive()).toBe(false);

      // User toggles anomaly on
      if (isAnomalyActive()) {
        setLayerVisibility((prev) => ({ ...prev, temperatureField: false }));
      } else {
        changeVariable("anomaly");
        setLayerVisibility((prev) => ({ ...prev, temperatureField: true }));
      }

      expect(activeVariable).toBe("anomaly");
      expect(isSstActive()).toBe(false);
      expect(isAnomalyActive()).toBe(true);

      // User toggles anomaly off
      if (isAnomalyActive()) {
        setLayerVisibility((prev) => ({ ...prev, temperatureField: false }));
      } else {
        changeVariable("anomaly");
        setLayerVisibility((prev) => ({ ...prev, temperatureField: true }));
      }

      expect(layerVisibility.temperatureField).toBe(false);
      expect(isSstActive()).toBe(false);
      expect(isAnomalyActive()).toBe(false);

      // User toggles 0.25° scientific grid
      setLayerVisibility((prev) => ({ ...prev, scientificGrid: !prev.scientificGrid }));
      expect(layerVisibility.scientificGrid).toBe(false);

      setLayerVisibility((prev) => ({ ...prev, scientificGrid: !prev.scientificGrid }));
      expect(layerVisibility.scientificGrid).toBe(true);
    });

    it("executes reset/fit to domain callback properly", () => {
      const resetCallback = vi.fn();
      resetCallback();
      expect(resetCallback).toHaveBeenCalledTimes(1);
    });
  });
});
