const EXPECTED_DIMENSION = 256;

export const EMBEDDING_EXPECTED_DIMENSION = EXPECTED_DIMENSION;
export const EMBEDDING_COLOR_MODES = Object.freeze([
  "month", "latitude", "longitude", "anomaly", "region",
]);

/** Normalize only fields present in the backend contract; never infer location or anomaly. */
export function normalizeEmbeddingResponse(raw, requestedDate) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return { ok: false, reason: "GET /embedding returned an invalid response." };
  }
  if (raw.date !== requestedDate) {
    return { ok: false, reason: `GET /embedding returned ${String(raw.date)} for requested date ${requestedDate}.` };
  }
  const declaredDim = Number.isInteger(raw.dim) && raw.dim > 0 ? raw.dim : null;
  const vector = raw.embedding;
  if (!Array.isArray(vector) || !declaredDim || vector.length !== declaredDim) {
    return { ok: false, reason: "The backend embedding vector and declared dimension do not match." };
  }
  if (!vector.every((value) => typeof value === "number" && Number.isFinite(value))) {
    return { ok: false, reason: "The embedding contains non-finite or non-numeric values." };
  }
  return {
    ok: true,
    value: {
      id: typeof raw.id === "string" || Number.isInteger(raw.id) ? String(raw.id) : requestedDate,
      date: requestedDate,
      embedding: vector,
      dimension: declaredDim,
      dataMode: typeof raw.data_mode === "string" ? raw.data_mode : null,
      month: Number.isInteger(raw.month) && raw.month >= 1 && raw.month <= 12 ? raw.month : monthOfDate(requestedDate),
      latitude: typeof raw.latitude === "number" && Number.isFinite(raw.latitude) ? raw.latitude : null,
      longitude: typeof raw.longitude === "number" && Number.isFinite(raw.longitude) ? raw.longitude : null,
      anomaly: typeof raw.anomaly === "number" && Number.isFinite(raw.anomaly) ? raw.anomaly : null,
      region: typeof raw.region === "string" && raw.region.trim() ? raw.region.trim() : null,
    },
  };
}

export function normalizeEmbeddingDataset(rawItems, requestedDates) {
  if (!Array.isArray(rawItems) || !Array.isArray(requestedDates) || rawItems.length !== requestedDates.length) {
    return { ok: false, reason: "The embedding responses are incomplete." };
  }
  const records = [];
  const invalid = [];
  rawItems.forEach((raw, index) => {
    const result = normalizeEmbeddingResponse(raw, requestedDates[index]);
    if (result.ok) records.push(result.value);
    else invalid.push({ date: requestedDates[index], reason: result.reason });
  });
  const dimensions = [...new Set(records.map((item) => item.dimension))];
  const counts = new Map(dimensions.map((dimension) => [dimension, records.filter((item) => item.dimension === dimension).length]));
  const dimension = dimensions.sort((a, b) => counts.get(b) - counts.get(a))[0] ?? null;
  const consistent = dimension !== null;
  const selectedDimensionRecords = records.filter((item) => item.dimension === dimension);
  return {
    ok: true,
    records: selectedDimensionRecords,
    dimension,
    invalidCount: invalid.length + records.length - selectedDimensionRecords.length,
    invalidReasons: invalid,
    warning: dimensions.length > 1 ? `Backend returned inconsistent embedding dimensions (${dimensions.join(", ")}); observations outside the most common dimension were excluded.` : null,
    expectedDimensionMismatch: consistent && dimension !== EXPECTED_DIMENSION,
  };
}

export function monthOfDate(date) {
  if (typeof date !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(date)) return null;
  const month = Number(date.slice(5, 7));
  return month >= 1 && month <= 12 ? month : null;
}

export function getEmbeddingColor(record, mode) {
  if (mode === "month") return Number.isInteger(record.month) ? record.month : monthOfDate(record.date);
  if (mode === "latitude") return Number.isFinite(record.latitude) ? record.latitude : null;
  if (mode === "longitude") return Number.isFinite(record.longitude) ? record.longitude : null;
  if (mode === "anomaly") return Number.isFinite(record.anomaly) ? record.anomaly : null;
  if (mode === "region") return typeof record.region === "string" && record.region ? record.region : null;
  return null;
}

export function colorDomain(records, mode) {
  const values = records.map((record) => getEmbeddingColor(record, mode)).filter((v) => v !== null);
  if (!values.length) return null;
  if (mode === "month" || mode === "region") return [...new Set(values)].sort((a, b) => a - b);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const extent = mode === "anomaly" ? Math.max(Math.abs(min), Math.abs(max)) : null;
  return mode === "anomaly" ? [-extent, extent] : [min, max];
}

export function monthColor(month) {
  return month == null ? "#64748b" : `hsl(${((month - 1) * 30 + 185) % 360} 72% 56%)`;
}

export function continuousColor(value, domain, mode) {
  if (!Number.isFinite(value) || !domain) return "#64748b";
  const [min, max] = domain;
  const t = min === max ? 0.5 : Math.min(1, Math.max(0, (value - min) / (max - min)));
  if (mode === "anomaly") {
    const hue = t < 0.5 ? 215 : 8;
    const light = 68 - Math.abs(t - 0.5) * 42;
    return `hsl(${hue} 78% ${light}%)`;
  }
  return `hsl(${195 - t * 150} 72% ${62 - t * 12}%)`;
}

export function regionColor(region, categories) {
  const index = categories.indexOf(region);
  return index < 0 ? "#64748b" : `hsl(${(index * 137.508 + 185) % 360} 68% 58%)`;
}
