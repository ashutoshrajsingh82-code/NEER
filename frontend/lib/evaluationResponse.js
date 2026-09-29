import { DEPTH_LEVELS } from "./oceanDomain.js";

export const EVALUATION_MODELS = Object.freeze([
  { key: "neer", label: "NEER", color: "#22d3ee" },
  { key: "climatology", label: "Climatology", color: "#a78bfa" },
  { key: "ridge", label: "Ridge", color: "#34d399" },
  { key: "lightgbm", label: "LightGBM", color: "#fbbf24" },
]);
export const EVALUATION_METRICS = Object.freeze([
  { key: "rmse", title: "RMSE", unit: true },
  { key: "mae", title: "MAE", unit: true },
  { key: "bias", title: "Bias", unit: true },
  { key: "pearson", title: "Pearson correlation", unit: false },
]);

const finiteOrNull = (value) => value === null || value === undefined || (typeof value === "number" && Number.isFinite(value));
const metricInRange = (metric, value) => metric === "pearson" ? value >= -1 && value <= 1 : (metric === "rmse" || metric === "mae") ? value >= 0 : true;
const depthValue = (label) => {
  if (typeof label === "number" && Number.isFinite(label)) return label;
  const match = String(label ?? "").match(/^\s*(-?\d+(?:\.\d+)?)\s*m?\s*$/i);
  return match ? Number(match[1]) : null;
};

export function validateEvaluationResponse(raw, expectedSplit = "test") {
  if (!raw || typeof raw !== "object" || raw.split !== expectedSplit || !raw.models || typeof raw.models !== "object") {
    return { ok: false, reason: "Evaluation response is missing its split or model results." };
  }
  if (typeof raw.is_synthetic !== "boolean" || typeof raw.data_mode !== "string" || !Array.isArray(raw.depths)) {
    return { ok: false, reason: "Evaluation source, data mode, or depth metadata is invalid." };
  }
  const depths = [...new Set(raw.depths.map(depthValue).filter((depth) => Number.isFinite(depth) && DEPTH_LEVELS.includes(depth)))]
    .sort((a, b) => a - b);
  const invalidDepths = raw.depths.length !== depths.length;
  const invalidValues = [];
  const models = {};
  for (const { key } of EVALUATION_MODELS) {
    const source = raw.models[key];
    if (source == null) { models[key] = null; continue; }
    if (typeof source !== "object" || !source.per_depth || typeof source.per_depth !== "object") {
      models[key] = null;
      invalidValues.push(`${key}: malformed metric block`);
      continue;
    }
    const perDepth = {};
    for (const depth of depths) {
      const label = Object.keys(source.per_depth).find((candidate) => depthValue(candidate) === depth);
      const value = label === undefined ? null : source.per_depth[label];
      if (value == null) { perDepth[depth] = null; continue; }
      if (typeof value !== "object") { perDepth[depth] = null; invalidValues.push(`${key} ${depth}m`); continue; }
      const clean = {};
      for (const { key: metric } of EVALUATION_METRICS) {
        const candidate = value[metric];
        if (finiteOrNull(candidate) && (candidate == null || metricInRange(metric, candidate))) clean[metric] = candidate ?? null;
        else { clean[metric] = null; invalidValues.push(`${key} ${depth}m ${metric}`); }
      }
      perDepth[depth] = clean;
    }
    const overall = source.overall && typeof source.overall === "object" ? source.overall : null;
    models[key] = { per_depth: perDepth, overall };
  }
  return { ok: true, value: {
    ...raw,
    source: raw.is_synthetic ? "synthetic_demo" : "test_reanalysis",
    depths,
    models,
    invalidDepths,
    invalidValues,
  } };
}

export function validateArgoEvaluationResponse(raw, { allowDemo = false } = {}) {
  if (!raw || typeof raw !== "object") return { ok: false, reason: "ARGO validation returned no usable response." };
  const isDemo = raw.validation_type === "DEMO_PIPELINE_CHECK_NOT_OBSERVATIONAL";
  if (isDemo && !allowDemo) {
    return { ok: false, reason: "The ARGO endpoint returned a demo pipeline check, not independent observations." };
  }
  if (!isDemo && raw.observational_validation !== true) {
    return { ok: false, reason: "The ARGO endpoint did not return independent observational validation." };
  }
  const metrics = isDemo ? raw.pipeline_check_metrics : raw.metrics;
  if (!metrics || typeof metrics !== "object" || !metrics.per_depth || typeof metrics.per_depth !== "object") {
    return { ok: false, reason: "Independent ARGO metrics are unavailable in this response." };
  }
  const gridDepths = Array.isArray(raw.neer_grid?.depth_m) ? raw.neer_grid.depth_m.map(Number) : [];
  const depths = [...new Set(gridDepths.filter((depth) => Number.isFinite(depth) && DEPTH_LEVELS.includes(depth)))].sort((a, b) => a - b);
  const perDepth = {};
  const invalidValues = [];
  for (const depth of depths) {
    const label = Object.keys(metrics.per_depth).find((candidate) => depthValue(candidate) === depth);
    const block = label === undefined ? null : metrics.per_depth[label];
    perDepth[depth] = {};
    for (const { key } of EVALUATION_METRICS) {
      const metric = key === "pearson" ? "correlation" : key;
      const value = block?.[metric];
      if (finiteOrNull(value) && (value == null || metricInRange(key, value))) perDepth[depth][key] = value ?? null;
      else { perDepth[depth][key] = null; invalidValues.push(`${depth}m ${metric}`); }
    }
  }
  return { ok: true, value: { ...raw, metrics, depths, per_depth: perDepth, invalidValues, isDemo } };
}
