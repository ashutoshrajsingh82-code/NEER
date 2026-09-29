import { DEPTH_LEVELS, OCEAN_DOMAIN } from "./oceanDomain.js";
import { isIsoDate } from "./dateDepthModel.js";
import { divergingColor } from "./colorScale.js";

export function validateHovmollerSelection({ lat, lon, startDate, endDate, minDepth, maxDepth, availableDates, availableDepths }) {
  if (!Number.isFinite(lat) || lat < OCEAN_DOMAIN.latMin || lat > OCEAN_DOMAIN.latMax) return { ok: false, field: "latitude", reason: `Latitude must be between ${OCEAN_DOMAIN.latMin} and ${OCEAN_DOMAIN.latMax}.` };
  if (!Number.isFinite(lon) || lon < OCEAN_DOMAIN.lonMin || lon > OCEAN_DOMAIN.lonMax) return { ok: false, field: "longitude", reason: `Longitude must be between ${OCEAN_DOMAIN.lonMin} and ${OCEAN_DOMAIN.lonMax}.` };
  if (!isIsoDate(startDate) || !availableDates.includes(startDate)) return { ok: false, field: "startDate", reason: "Choose a start date available from the backend." };
  if (!isIsoDate(endDate) || !availableDates.includes(endDate)) return { ok: false, field: "endDate", reason: "Choose an end date available from the backend." };
  if (startDate > endDate) return { ok: false, field: "dateRange", reason: "Start date must be before or equal to end date." };
  const depths = availableDepths?.length ? availableDepths : DEPTH_LEVELS;
  if (!depths.includes(minDepth) || !depths.includes(maxDepth)) return { ok: false, field: "depthRange", reason: "Choose depth levels supported by the model." };
  if (minDepth > maxDepth) return { ok: false, field: "depthRange", reason: "Minimum depth must be less than or equal to maximum depth." };
  return { ok: true };
}

export function normalizeHovmoller(profiles, selectedDates, requested) {
  if (!Array.isArray(profiles) || profiles.length !== selectedDates.length || !profiles.length) return { ok: false, reason: "The backend returned an empty or incomplete time series." };
  const depths = profiles[0]?.depths;
  if (!Array.isArray(depths) || !depths.length || !Array.isArray(profiles[0]?.temperature) || profiles[0].temperature.length !== depths.length) return { ok: false, reason: "The backend returned a malformed profile." };
  if (depths.some((depth, index) => typeof depth !== "number" || !Number.isFinite(depth) || (index > 0 && depth <= depths[index - 1]))) return { ok: false, reason: "The backend returned an invalid depth axis." };
  if (selectedDates.some((date, index) => index > 0 && date <= selectedDates[index - 1])) return { ok: false, reason: "The backend dates are not in increasing time order." };
  for (let i = 0; i < profiles.length; i += 1) {
    const item = profiles[i];
    const optionalSeriesValid = (series) => series == null || (Array.isArray(series) && series.length === depths.length
      && series.every((v) => v === null || (typeof v === "number" && Number.isFinite(v))));
    const fieldsAligned = depths.every((_, index) => {
      const prediction = item.temperature?.[index];
      const climatology = item.climatology?.[index] ?? null;
      const anomaly = item.anomaly?.[index] ?? null;
      const valid = [prediction, climatology, anomaly].every((v) => typeof v === "number" && Number.isFinite(v));
      const missing = [prediction, climatology, anomaly].every((v) => v === null);
      const referenceUnavailable = typeof prediction === "number" && Number.isFinite(prediction)
        && climatology === null && anomaly === null;
      return (valid && Math.abs(anomaly - (prediction - climatology)) <= 1e-5) || missing || referenceUnavailable;
    });
    if (item.date !== selectedDates[i] || item.lat !== requested.lat || item.lon !== requested.lon
      || !Array.isArray(item.depths) || item.depths.length !== depths.length
      || item.depths.some((d, j) => d !== depths[j])
      || !Array.isArray(item.temperature) || item.temperature.length !== depths.length
      || item.temperature.some((v) => v !== null && (typeof v !== "number" || !Number.isFinite(v)))
      || !optionalSeriesValid(item.anomaly) || !optionalSeriesValid(item.climatology) || !fieldsAligned) {
      return { ok: false, reason: "The backend returned a mismatched or malformed profile." };
    }
  }
  const indices = depths.map((value, index) => ({ value, index })).filter(({ value }) => value >= requested.minDepth && value <= requested.maxDepth);
  if (!indices.length) return { ok: false, reason: "The backend returned no values in the selected depth range." };
  const selectSeries = (key) => profiles.map((item) => indices.map(({ index }) => item[key]?.[index] ?? null));
  return { ok: true, value: { lat: requested.lat, lon: requested.lon, startDate: requested.startDate, endDate: requested.endDate,
    dates: profiles.map((item) => item.date), depths: indices.map(({ value }) => value),
    values: selectSeries("temperature"), temperatureValues: selectSeries("temperature"), anomalyValues: selectSeries("anomaly"), climatologyValues: selectSeries("climatology"),
    dataMode: [...new Set(profiles.map((item) => item.data_mode).filter(Boolean))].join(", ") || "N/A" } };
}

export function hovmollerValues(data, variableMode) {
  return variableMode === "anomaly" ? data?.anomalyValues ?? null : data?.temperatureValues ?? data?.values ?? null;
}

export function hovmollerColor(value, min, max, variableMode = "temperature") {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  if (variableMode === "anomaly") {
    const limit = Math.max(Math.abs(min), Math.abs(max)) || 1;
    return divergingColor(Math.max(-1, Math.min(1, value / limit)));
  }
  const t = max === min ? 0.5 : Math.max(0, Math.min(1, (value - min) / (max - min)));
  const stops = [[22, 78, 150], [34, 211, 238], [250, 204, 21], [239, 68, 68]];
  const scaled = t * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const f = scaled - index;
  const rgb = stops[index].map((channel, i) => Math.round(channel + (stops[index + 1][i] - channel) * f));
  return `rgb(${rgb.join(",")})`;
}

export function hovmollerCsv(data) {
  const rows = [`# latitude,${data.lat}`, `# longitude,${data.lon}`, `# start_date,${data.startDate}`, `# end_date,${data.endDate}`, `# variable,${data.variableMode ?? "temperature"}`, `# definition,${data.variableMode === "anomaly" ? "model_output - climatology" : "model_output"}`, "date,depth_m,value_c"];
  data.dates.forEach((date, x) => data.depths.forEach((depth, y) => {
    const value = data.values[x][y];
    rows.push(`${date},${depth},${typeof value === "number" && Number.isFinite(value) ? value : ""}`);
  }));
  return rows.join("\n");
}
