import { describe, expect, it } from "vitest";
import { EXPLAINABILITY_FEATURE_ORDER, validateExplainabilityResponse } from "./explainabilityResponse.js";

const expected = { date: "2020-01-15", lat: 12.3, lon: 54.6, depth: 100 };

function validResponse(overrides = {}) {
  return {
    method: "Integrated Gradients",
    date: expected.date,
    lat: expected.lat,
    lon: expected.lon,
    depth: expected.depth,
    predicted_model_output: 2.5,
    baseline_model_output: 1.5,
    baseline: "Physical channels set to reference values; context fixed.",
    feature_order: [...EXPLAINABILITY_FEATURE_ORDER],
    features: EXPLAINABILITY_FEATURE_ORDER.map((name, index) => ({ name, attribution: index === 0 ? 0 : index === 1 ? -0.25 : 0.25 })),
    attribution_sum: 1,
    output_delta_from_baseline: 1,
    completeness_error: 0,
    temperature: 2.5,
    climatology: null,
    anomaly: null,
    climatology_lat: null,
    climatology_lon: null,
    ...overrides,
  };
}

describe("validateExplainabilityResponse", () => {
  it("preserves zero and signed attributions and fills missing channels as unavailable", () => {
    const raw = validResponse({ features: [
      { name: "sst", attribution: 0 },
      { name: "sss", attribution: -0.25 },
      { name: "sla", attribution: 1.25 },
    ] });
    const result = validateExplainabilityResponse(raw, expected);
    expect(result.ok).toBe(true);
    expect(result.value.features[0].attribution).toBe(0);
    expect(result.value.features[1].attribution).toBe(-0.25);
    expect(result.value.features[3].attribution).toBeNull();
    expect(result.value.partial).toBe(true);
  });

  it("rejects stale selection metadata and noncanonical feature order", () => {
    expect(validateExplainabilityResponse(validResponse({ lat: 12.4 }), expected).ok).toBe(false);
    expect(validateExplainabilityResponse(validResponse({ feature_order: [...EXPLAINABILITY_FEATURE_ORDER].reverse() }), expected).ok).toBe(false);
  });

  it("rejects inconsistent prediction context and completeness metadata", () => {
    expect(validateExplainabilityResponse(validResponse({ temperature: 5, climatology: 2, anomaly: 2.5 }), expected).ok).toBe(false);
    expect(validateExplainabilityResponse(validResponse({ attribution_sum: 2 }), expected).ok).toBe(false);
  });

  it("accepts decoder delta responses and model temperature without climatology", () => {
    expect(validateExplainabilityResponse(validResponse(), expected).ok).toBe(true);
    const currentBackendResponse = { ...validResponse() };
    delete currentBackendResponse.predicted_model_output;
    delete currentBackendResponse.baseline_model_output;
    expect(validateExplainabilityResponse(currentBackendResponse, expected).ok).toBe(true);
    expect(validateExplainabilityResponse({ ...currentBackendResponse, output_delta_from_baseline: null }, expected).ok).toBe(false);
  });
});
