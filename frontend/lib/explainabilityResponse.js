import { isIsoDate } from "./dateDepthModel.js";

export const EXPLAINABILITY_FEATURE_ORDER = Object.freeze([
  "sst", "sss", "sla", "u_current", "v_current", "u_wind", "v_wind",
]);

export const EXPLAINABILITY_FEATURES = Object.freeze({
  sst: { label: "SST", description: "Sea Surface Temperature" },
  sss: { label: "SSS", description: "Sea Surface Salinity" },
  sla: { label: "SLA", description: "Sea Level Anomaly" },
  u_current: { label: "U-current", description: "Eastward surface current component" },
  v_current: { label: "V-current", description: "Northward surface current component" },
  u_wind: { label: "U-wind", description: "Eastward surface wind component" },
  v_wind: { label: "V-wind", description: "Northward surface wind component" },
});

const finite = (value) => typeof value === "number" && Number.isFinite(value);
const nullableFinite = (value) => value === null || finite(value);

export function validateExplainabilityResponse(raw, expected) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return { ok: false, reason: "The explainability endpoint returned no usable response." };
  }
  if (raw.method !== "Integrated Gradients") {
    return { ok: false, reason: "The backend did not identify the method as Integrated Gradients." };
  }
  if (!isIsoDate(raw.date) || raw.date !== expected.date) {
    return { ok: false, reason: "The explanation date does not match the current selection." };
  }
  if (!finite(raw.lat) || !finite(raw.lon) || Math.abs(raw.lat - expected.lat) > 1e-6 || Math.abs(raw.lon - expected.lon) > 1e-6) {
    return { ok: false, reason: "The explanation location does not match the current selection." };
  }
  if (!finite(raw.depth) || Math.abs(raw.depth - expected.depth) > 1e-6) {
    return { ok: false, reason: "The explanation depth does not match the current selection." };
  }
  const hasLegacyOutputs = finite(raw.predicted_model_output) && finite(raw.baseline_model_output);
  // Current backend contract reports the decoder's output change from its
  // baseline, not both absolute decoder values.
  if (!hasLegacyOutputs && !finite(raw.output_delta_from_baseline)) {
    return { ok: false, reason: "The backend decoder output change from baseline is missing or non-finite." };
  }
  if (!Array.isArray(raw.features) || !Array.isArray(raw.feature_order) || JSON.stringify(raw.feature_order) !== JSON.stringify(EXPLAINABILITY_FEATURE_ORDER)) {
    return { ok: false, reason: "The backend feature order is missing or does not match NEER's canonical input channels." };
  }
  const byName = new Map();
  let previousIndex = -1;
  for (const feature of raw.features) {
    const index = feature ? EXPLAINABILITY_FEATURE_ORDER.indexOf(feature.name) : -1;
    if (index < 0 || byName.has(feature.name) || index <= previousIndex) {
      return { ok: false, reason: "The backend returned an unknown or duplicate feature name." };
    }
    previousIndex = index;
    if (feature.attribution !== null && feature.attribution !== undefined && !finite(feature.attribution)) {
      return { ok: false, reason: "A feature attribution is non-finite." };
    }
    byName.set(feature.name, feature);
  }
  const features = EXPLAINABILITY_FEATURE_ORDER.map((name) => ({
    name,
    label: EXPLAINABILITY_FEATURES[name].label,
    description: EXPLAINABILITY_FEATURES[name].description,
    attribution: byName.has(name) ? byName.get(name).attribution ?? null : null,
  }));
  const missingFeatures = features.filter((feature) => feature.attribution === null).map((feature) => feature.name);
  const nullableFields = ["temperature", "climatology", "anomaly", "climatology_lat", "climatology_lon"];
  if (nullableFields.some((field) => !nullableFinite(raw[field]))) {
    return { ok: false, reason: "The backend returned a non-finite prediction or location value." };
  }
  const hasTemperature = finite(raw.temperature);
  const hasClimatology = finite(raw.climatology);
  const hasAnomaly = finite(raw.anomaly);
  if (raw.temperature === null && (raw.climatology !== null || raw.anomaly !== null)) {
    return { ok: false, reason: "Climatology and anomaly cannot be present when model temperature is unavailable." };
  }
  if (hasTemperature && (hasClimatology !== hasAnomaly)) {
    return { ok: false, reason: "Climatology and anomaly must either both be present or both unavailable." };
  }
  if (hasTemperature && hasClimatology && Math.abs(raw.temperature - raw.climatology - raw.anomaly) > 1e-5) {
    return { ok: false, reason: "The returned temperature does not match the canonical anomaly and climatology." };
  }
  if (!finite(raw.completeness_error) || !finite(raw.attribution_sum) || !finite(raw.output_delta_from_baseline)) {
    return { ok: false, reason: "Integrated Gradients completeness metadata is missing or non-finite." };
  }
  if (Math.abs(raw.attribution_sum - raw.output_delta_from_baseline - raw.completeness_error) > 1e-5) {
    return { ok: false, reason: "The returned attribution completeness values are inconsistent." };
  }
  if (typeof raw.baseline !== "string" || !raw.baseline) {
    return { ok: false, reason: "The backend did not describe the Integrated Gradients baseline." };
  }
  return {
    ok: true,
    value: {
      ...raw,
      features,
      context_attributions: Array.isArray(raw.context_attributions) ? raw.context_attributions : [],
      partial: missingFeatures.length > 0,
      missingFeatures,
    },
  };
}
