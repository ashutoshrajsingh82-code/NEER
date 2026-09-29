/** Helpers for displaying authoritative NEER anomaly data without deriving it in the UI. */

export const isFiniteScientificValue = (value) => typeof value === "number" && Number.isFinite(value);

/** Summarizes actual finite values in a nested grid using one traversal. */
export function summarizeFiniteValues(values) {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let validCells = 0;
  const visit = (value) => {
    if (Array.isArray(value)) {
      value.forEach(visit);
    } else if (isFiniteScientificValue(value)) {
      min = Math.min(min, value);
      max = Math.max(max, value);
      sum += value;
      validCells += 1;
    }
  };
  visit(values);
  return validCells
    ? { min, max, mean: sum / validCells, validCells }
    : { min: null, max: null, mean: null, validCells: 0 };
}

/** Validate one-depth /reconstruct/grid data and its scientific coordinates. */
export function validateGridAnomalyResponse(data, expected) {
  const finiteAxis = (axis, min, max) => Array.isArray(axis) && axis.length > 0
    && axis.every((value, index) => Number.isFinite(value) && value >= min && value <= max && (index === 0 || value > axis[index - 1]));
  if (data?.date !== expected.date || data?.depth !== expected.depth
    || !finiteAxis(data?.lat, expected.latMin, expected.latMax)
    || !finiteAxis(data?.lon, expected.lonMin, expected.lonMax)) return { ok: false, reason: "The backend grid coordinates do not match the requested scientific context." };
  const gridMatchesAxes = (grid) => Array.isArray(grid) && grid.length === data.lat.length
    && grid.every((row) => Array.isArray(row) && row.length === data.lon.length
      && row.every((value) => value === null || (typeof value === "number" && Number.isFinite(value))));
  if (!gridMatchesAxes(data.temperature) || !gridMatchesAxes(data.anomaly)
    || (data.climatology != null && !gridMatchesAxes(data.climatology))) {
    return { ok: false, reason: "The backend returned malformed or misaligned grid fields." };
  }
  if (data.climatology == null && data.anomaly.some((row) => row.some((value) => value !== null))) {
    return { ok: false, reason: "The backend returned anomaly values without climatology." };
  }
  const valuesAligned = data.temperature.every((row, i) => row.every((prediction, j) => {
    const climatology = data.climatology?.[i]?.[j] ?? null;
    const anomaly = data.anomaly[i][j];
    const valid = [prediction, climatology, anomaly].every((value) => typeof value === "number" && Number.isFinite(value));
    const missing = [prediction, climatology, anomaly].every((value) => value === null);
    const referenceUnavailable = typeof prediction === "number" && Number.isFinite(prediction)
      && climatology === null && anomaly === null;
    return (valid && Math.abs(anomaly - (prediction - climatology)) <= 1e-5) || missing || referenceUnavailable;
  }));
  if (!valuesAligned) return { ok: false, reason: "The backend grid fields do not satisfy model output minus climatology." };
  return { ok: true, value: data };
}

export function formatTemperature(value, digits = 2) {
  return isFiniteScientificValue(value) ? `${value.toFixed(digits)} °C` : "N/A";
}

export function formatAnomaly(value, digits = 2) {
  if (!isFiniteScientificValue(value)) return "N/A";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)} °C`;
}
