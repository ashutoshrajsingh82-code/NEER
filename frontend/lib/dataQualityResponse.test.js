import { describe, expect, it } from "vitest";
import { validateDataQuality } from "./dataQualityResponse.js";

function payload(overrides = {}) {
  return {
    data_mode: "OBSERVED", is_synthetic: false, source_tensors_path: "/data/tensors.npz",
    dates: { count: 2, min_date: "2024-01-01", max_date: "2024-01-05" },
    summary: { inputs: { valid_fraction: 0.5 } },
    spatial_coverage: { grid_shape: [2, 2], lat_range: [5, 6], lon_range: [45, 46], by_period: {} },
    channels: { sst: { description: "Sea surface temperature", n_cells: 8, n_valid: 4, n_missing: 4, valid_fraction: 0.5 } },
    input_cells: { n_cells: 16, n_valid: 8, n_missing: 8, scope: "all input data" },
    observed_input_cells: { n_cells: 8, n_valid: 4, n_missing: 4, scope: "physical channels" },
    coverage_grid: { lat: [5, 6], lon: [45, 46], cells: [[1, 0], [-1, 1]], valid_cells: 2, missing_cells: 1, outside_cells: 1, scope: "actual masks" },
    grid_metadata: { lat_points: 2, lon_points: 2, lat_resolution: { uniform: true, degrees: 1 }, lon_resolution: { uniform: true, degrees: 1 }, target_depths: [0, 100] },
    provenance: { source: "source.nc", dataset_format: "NetCDF", version: "1.0", preprocessing_pipeline: "Pipeline", temporal_frequency: "monthly", variables: { sst: { units: "degC" } } },
    targets: { variables: ["temp"], n_cells: 8, n_valid: 6, valid_fraction: 0.75, per_depth: { "0.0": { n_cells: 4, n_valid: 3, valid_fraction: 0.75 }, "100.0": { n_cells: 4, n_valid: 3, valid_fraction: 0.75 } } },
    ...overrides,
  };
}

describe("validateDataQuality", () => {
  it("keeps backend dates, units, actual missing counts, and partial status", () => {
    const result = validateDataQuality(payload(), { dates: ["2024-01-01", "2024-01-05"], isSynthetic: false }, { architecture: { depths: [0, 100] } });
    expect(result.ok).toBe(true);
    expect(result.value.dates.dates).toEqual(["2024-01-01", "2024-01-05"]);
    expect(result.value.quality_status).toBe("PARTIAL");
    expect(result.value.target_by_depth[100].n_missing).toBe(1);
    expect(result.value.channel_rows[0].units).toBe("degC");
  });

  it("rejects bad counts, inconsistent grid dimensions, and non-finite numbers", () => {
    expect(validateDataQuality(payload({ observed_input_cells: { n_cells: 8, n_valid: 9, n_missing: -1 } }), { dates: [] }).ok).toBe(false);
    expect(validateDataQuality(payload({ coverage_grid: { lat: [5], lon: [45], cells: [[1]], valid_cells: 1, missing_cells: 0 } }), { dates: [] }).ok).toBe(false);
    expect(validateDataQuality(payload({ summary: { inputs: { valid_fraction: Infinity } } }), { dates: [] }).ok).toBe(false);
  });

  it("allows date retrieval failure while retaining backend dataset-level metadata", () => {
    const result = validateDataQuality(payload(), null);
    expect(result.ok).toBe(true);
    expect(result.value.dates.dates).toBeNull();
    expect(result.value.dates.count).toBe(2);
  });
});
