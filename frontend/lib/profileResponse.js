/** Data-shape and context checks for GET /profile. */
export function validateProfileResponse(data, expected) {
  const arraysValid = Array.isArray(data?.depths) && Array.isArray(data?.temperature) && Array.isArray(data?.anomaly)
    && data.depths.length > 0 && data.depths.length === data.temperature.length && data.depths.length === data.anomaly.length
    && data.depths.every((v) => typeof v === "number" && Number.isFinite(v))
    && data.temperature.every((v) => v === null || (typeof v === "number" && Number.isFinite(v)))
    && data.anomaly.every((v) => v === null || (typeof v === "number" && Number.isFinite(v)));
  const climatologyValid = data?.climatology == null || (
    Array.isArray(data.climatology) && data.climatology.length === data.depths?.length
      && data.climatology.every((v) => v === null || (typeof v === "number" && Number.isFinite(v)))
  );
  const fieldsAligned = arraysValid && data.depths.every((_, index) => {
    const prediction = data.temperature[index];
    const climatology = data.climatology?.[index] ?? null;
    const anomaly = data.anomaly[index];
    const allFinite = [prediction, climatology, anomaly].every((value) => typeof value === "number" && Number.isFinite(value));
    const allMissing = [prediction, climatology, anomaly].every((value) => value === null);
    const referenceUnavailable = typeof prediction === "number" && Number.isFinite(prediction)
      && climatology === null && anomaly === null;
    return (allFinite && Math.abs(anomaly - (prediction - climatology)) <= 1e-5) || allMissing || referenceUnavailable;
  });
  if (!arraysValid || !climatologyValid || !fieldsAligned || data.date !== expected.date || data.lat !== expected.lat || data.lon !== expected.lon) {
    return { ok: false, reason: "The backend returned an incomplete or mismatched profile." };
  }
  return { ok: true, value: data };
}

/** Render a present scientific value, including zero, or N/A for absent/invalid values. */
export function valueOrNA(value) {
  if (value === null || value === undefined) return "N/A";
  if (typeof value === "number" && !Number.isFinite(value)) return "N/A";
  if (typeof value === "string" && value.trim() === "") return "N/A";
  return ["string", "number", "boolean"].includes(typeof value) ? value : "N/A";
}

export function profileSeries(data) {
  const hasValues = (values) => Array.isArray(values) && values.some((value) => typeof value === "number" && Number.isFinite(value));
  return [
    { key: "neer", label: "MODEL OUTPUT", values: data?.temperature, color: "#2dd4bf", dash: "" },
    { key: "climatology", label: "CLIMATOLOGY", values: data?.climatology, color: "#fbbf24", dash: "7 4" },
    { key: "anomaly", label: "ANOMALY", values: data?.anomaly, color: "#fb7185", dash: "" },
    { key: "argo", label: "ARGO", values: null, color: "#c084fc", dash: "2 4" },
  ].map((series) => ({ ...series, available: hasValues(series.values) }));
}

export function profilePointAtDepth(data, depth) {
  const index = data?.depths?.findIndex((value) => value === depth) ?? -1;
  if (index < 0) return null;
  return {
    depth,
    neer: valueOrNA(data.temperature?.[index]),
    climatology: valueOrNA(data.climatology?.[index]),
    anomaly: valueOrNA(data.anomaly?.[index]),
    argo: "N/A",
  };
}

export function profileCsv(data) {
  const rows = [
    `# latitude,${data.lat}`,
    `# longitude,${data.lon}`,
    `# date,${data.date}`,
    `# model_version,${valueOrNA(data.model_version ?? data.modelVersion)}`,
    `# data_mode,${valueOrNA(data.data_mode)}`,
    "depth_m,model_output_temperature_c,climatology_temperature_c,anomaly_c,argo_temperature_c",
  ];
  data.depths.forEach((depth, index) => {
    const cell = (value) => typeof value === "number" && Number.isFinite(value) ? String(value) : "";
    rows.push([depth, cell(data.temperature[index]), cell(data.climatology?.[index]), cell(data.anomaly?.[index]), ""].join(","));
  });
  return rows.join("\n");
}
