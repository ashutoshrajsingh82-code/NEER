import { describe, expect, it } from "vitest";
import { validateArgoEvaluationResponse, validateEvaluationResponse } from "./evaluationResponse.js";

const comparison = (overrides = {}) => ({
  split: "test", is_synthetic: false, data_mode: "REANALYSIS", depths: ["100m", "0m", "5m"],
  models: { neer: { overall: {}, per_depth: { "0m": { rmse: 1, mae: 0.5, bias: -0.1, pearson: 0.4 }, "5m": { rmse: null, mae: null, bias: null, pearson: null }, "100m": { rmse: 2, mae: 1, bias: 0.1, pearson: 1.2 } } }, climatology: null, ridge: null, lightgbm: null },
  ...overrides,
});

describe("evaluation response validation", () => {
  it("sorts available depths numerically and preserves missing metrics", () => {
    const result = validateEvaluationResponse(comparison());
    expect(result.ok).toBe(true);
    expect(result.value.depths).toEqual([0, 5, 100]);
    expect(result.value.models.neer.per_depth[5].rmse).toBeNull();
    expect(result.value.models.ridge).toBeNull();
    expect(result.value.models.neer.per_depth[0].bias).toBe(-0.1);
    expect(result.value.models.neer.per_depth[100].pearson).toBeNull();
  });

  it("keeps synthetic provenance distinct and rejects malformed responses", () => {
    expect(validateEvaluationResponse(comparison({ is_synthetic: true })).value.source).toBe("synthetic_demo");
    expect(validateEvaluationResponse({ split: "val" }, "test").ok).toBe(false);
  });

  it("accepts only explicit independent ARGO observational metrics", () => {
    const result = validateArgoEvaluationResponse({
      observational_validation: true, validation_type: "ARGO_PROFILE_VALIDATION",
      neer_grid: { depth_m: [0, 5] }, metrics: { overall: { rmse: 1, mae: 0.5, bias: -0.2, correlation: 0.7, n_pairs: 8 }, per_depth: { "0m": { rmse: 1, mae: 0.5, bias: -0.2, correlation: 0.7 }, "5m": { rmse: null, mae: null, bias: null, correlation: null } } },
    });
    expect(result.ok).toBe(true);
    expect(result.value.per_depth[0].pearson).toBe(0.7);
    expect(validateArgoEvaluationResponse({ observational_validation: false }).ok).toBe(false);
  });

  it("keeps the labelled synthetic ARGO pipeline check separate from observations", () => {
    const demo = {
      validation_type: "DEMO_PIPELINE_CHECK_NOT_OBSERVATIONAL", observational_validation: false,
      neer_grid: { depth_m: [0] }, pipeline_check_metrics: {
        overall: { rmse: 1, mae: 0.8, bias: 0.2, correlation: 0.5, n_pairs: 10 },
        per_depth: { "0m": { rmse: 1, mae: 0.8, bias: 0.2, correlation: 0.5 } },
      },
    };
    expect(validateArgoEvaluationResponse(demo).ok).toBe(false);
    const result = validateArgoEvaluationResponse(demo, { allowDemo: true });
    expect(result.ok).toBe(true);
    expect(result.value.isDemo).toBe(true);
    expect(result.value.metrics.overall.n_pairs).toBe(10);
  });
});
