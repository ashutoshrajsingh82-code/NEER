import { DEPTH_LEVELS, OCEAN_DOMAIN } from "./oceanDomain.js";

const finiteOrNull = (value) => value === null || (typeof value === "number" && Number.isFinite(value));

export function validateOceanVolume(data, expected) {
  const coordinates = (items, min, max) => Array.isArray(items) && items.length > 0 && items.every((value, i) => Number.isFinite(value) && value >= min && value <= max && (i === 0 || value > items[i - 1]));
  const depths = data?.depths;
  const latitudes = data?.lat;
  const longitudes = data?.lon;
  if (data?.date !== expected.date || !coordinates(latitudes, OCEAN_DOMAIN.latMin, OCEAN_DOMAIN.latMax)
    || !coordinates(longitudes, OCEAN_DOMAIN.lonMin, OCEAN_DOMAIN.lonMax)
    || !Array.isArray(depths) || !depths.length || depths.some((d, i) => !Number.isFinite(d) || (i > 0 && d <= depths[i - 1]))) {
    return { ok: false, reason: "The backend returned mismatched or invalid volume coordinates." };
  }
  const depthAllowed = expected.depths?.length ? expected.depths : DEPTH_LEVELS;
  if (depths.some((d) => !depthAllowed.includes(d))) return { ok: false, reason: "The backend returned an unsupported depth level." };
  const matrixValid = (matrix) => Array.isArray(matrix) && matrix.length === latitudes.length
    && matrix.every((row) => Array.isArray(row) && row.length === longitudes.length
      && row.every((column) => Array.isArray(column) && column.length === depths.length && column.every(finiteOrNull)));
  if (!matrixValid(data.temperature)) return { ok: false, reason: "The backend returned a malformed temperature volume." };
  if (data.climatology != null && !matrixValid(data.climatology)) return { ok: false, reason: "The backend returned a malformed climatology volume." };
  if (!matrixValid(data.anomaly)) return { ok: false, reason: "The backend returned a malformed or misaligned anomaly volume." };
  if (data.climatology == null && data.anomaly.some((row) => row.some((column) => column.some((value) => value !== null)))) return { ok: false, reason: "The backend returned anomaly values without climatology." };
  const fieldsAligned = data.temperature.every((row, i) => row.every((column, j) => column.every((prediction, k) => {
    const climatology = data.climatology?.[i]?.[j]?.[k] ?? null;
    const anomaly = data.anomaly[i][j][k];
    const valid = [prediction, climatology, anomaly].every((value) => typeof value === "number" && Number.isFinite(value));
    const missing = [prediction, climatology, anomaly].every((value) => value === null);
    const referenceUnavailable = typeof prediction === "number" && Number.isFinite(prediction)
      && climatology === null && anomaly === null;
    return (valid && Math.abs(anomaly - (prediction - climatology)) <= 1e-5) || missing || referenceUnavailable;
  })));
  if (!fieldsAligned) return { ok: false, reason: "The backend volume fields do not satisfy model output minus climatology." };
  if (expected.latMin !== undefined && (latitudes[0] < expected.latMin || latitudes.at(-1) > expected.latMax)) return { ok: false, reason: "The backend returned latitude coordinates outside the requested region." };
  if (expected.lonMin !== undefined && (longitudes[0] < expected.lonMin || longitudes.at(-1) > expected.lonMax)) return { ok: false, reason: "The backend returned longitude coordinates outside the requested region." };
  return { ok: true, value: data };
}

export function volumeValue(data, latitudeIndex, longitudeIndex, depthIndex, variable) {
  if (variable === "anomaly") return data.anomaly?.[latitudeIndex]?.[longitudeIndex]?.[depthIndex] ?? null;
  if (variable === "climatology") return data.climatology?.[latitudeIndex]?.[longitudeIndex]?.[depthIndex] ?? null;
  return data.temperature?.[latitudeIndex]?.[longitudeIndex]?.[depthIndex] ?? null;
}

export function summarizeVolumeDepthRange(data, variable, minDepth, maxDepth) {
  let min = Infinity; let max = -Infinity; let sum = 0; let validCells = 0;
  if (!Array.isArray(data?.depths)) return { min: null, max: null, mean: null, validCells: 0 };
  data.depths.forEach((depth, k) => {
    if (depth < minDepth || depth > maxDepth) return;
    data.lat.forEach((_, i) => data.lon.forEach((__, j) => {
      const value = volumeValue(data, i, j, k, variable);
      if (typeof value === "number" && Number.isFinite(value)) {
        min = Math.min(min, value); max = Math.max(max, value); sum += value; validCells += 1;
      }
    }));
  });
  return validCells ? { min, max, mean: sum / validCells, validCells } : { min: null, max: null, mean: null, validCells: 0 };
}

export function volumeColor(value, min, max, variable = "temperature") {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  const anomalyScale = Math.max(Math.abs(min), Math.abs(max), 1e-9);
  const t = variable === "anomaly"
    ? Math.max(0, Math.min(1, (value + anomalyScale) / (2 * anomalyScale)))
    : max === min ? 0.5 : Math.max(0, Math.min(1, (value - min) / (max - min)));
  const stops = variable === "anomaly" ? [[59,130,246],[230,237,245],[239,68,68]] : [[22,78,150],[34,211,238],[250,204,21],[239,68,68]];
  const s = t * (stops.length - 1); const i = Math.min(stops.length - 2, Math.floor(s)); const f = s - i;
  return stops[i].map((v, channel) => Math.round(v + (stops[i + 1][channel] - v) * f));
}
