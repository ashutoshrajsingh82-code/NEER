const finite = (value) => typeof value === "number" && Number.isFinite(value);
const nonnegativeInt = (value) => Number.isInteger(value) && value >= 0;
const cleanText = (value) => typeof value === "string" && value.trim() ? value : null;

export function validateDataQuality(raw, datesPayload, modelPayload = null) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return { ok: false, reason: "The data-quality endpoint returned no usable object." };
  if (typeof raw.data_mode !== "string" || typeof raw.is_synthetic !== "boolean" || typeof raw.source_tensors_path !== "string") {
    return { ok: false, reason: "Dataset identity metadata is missing or invalid." };
  }
  let dates = null;
  if (datesPayload != null) {
    if (!Array.isArray(datesPayload.dates) || datesPayload.dates.some((date) => typeof date !== "string")) return { ok: false, reason: "The available-date list is invalid." };
    dates = [...new Set(datesPayload.dates)].sort();
    if (typeof datesPayload.isSynthetic === "boolean" && datesPayload.isSynthetic !== raw.is_synthetic) return { ok: false, reason: "Dates and quality metadata refer to different dataset modes." };
  }

  const { summary, spatial_coverage: spatial, channels, targets, observed_input_cells: observed, input_cells: inputs, coverage_grid: grid } = raw;
  if (!summary?.inputs || !spatial || !channels || typeof channels !== "object" || !observed || !inputs || !grid) {
    return { ok: false, reason: "The backend omitted required scope, channel, or spatial coverage metadata." };
  }
  for (const counts of [observed, inputs]) {
    if (![counts.n_cells, counts.n_valid, counts.n_missing].every(nonnegativeInt) || counts.n_valid + counts.n_missing !== counts.n_cells) {
      return { ok: false, reason: "Cell counts are invalid or internally inconsistent." };
    }
  }
  if (!Array.isArray(grid.lat) || !Array.isArray(grid.lon) || !Array.isArray(grid.cells) || grid.cells.length !== grid.lat.length || grid.cells.some((row) => !Array.isArray(row) || row.length !== grid.lon.length || row.some((cell) => ![-1, 0, 1].includes(cell)))) {
    return { ok: false, reason: "The coverage grid is malformed." };
  }
  if (![...grid.lat, ...grid.lon].every(finite)) return { ok: false, reason: "The coverage grid has non-finite coordinates." };
  const shape = spatial.grid_shape;
  if (!Array.isArray(shape) || shape[0] !== grid.lat.length || shape[1] !== grid.lon.length) return { ok: false, reason: "Coverage coordinates do not match the backend grid dimensions." };
  const flattenedCells = grid.cells.flat();
  const validGridCells = flattenedCells.filter((cell) => cell === 1).length;
  const missingGridCells = flattenedCells.filter((cell) => cell === 0).length;
  const outsideGridCells = flattenedCells.filter((cell) => cell === -1).length;
  if (!nonnegativeInt(grid.valid_cells) || !nonnegativeInt(grid.missing_cells) || validGridCells !== grid.valid_cells || missingGridCells !== grid.missing_cells || (grid.outside_cells != null && (!nonnegativeInt(grid.outside_cells) || outsideGridCells !== grid.outside_cells))) return { ok: false, reason: "Coverage map cells do not match the backend coverage counts." };
  for (const field of ["lat_range", "lon_range"]) {
    const range = spatial[field];
    const axis = field === "lat_range" ? grid.lat : grid.lon;
    if (range != null && (!Array.isArray(range) || range.length !== 2 || !range.every(finite))) return { ok: false, reason: `The spatial ${field} metadata is invalid.` };
    if (range && (Math.abs(Math.min(...axis) - Math.min(...range)) > 1e-6 || Math.abs(Math.max(...axis) - Math.max(...range)) > 1e-6)) return { ok: false, reason: `The spatial ${field} does not match grid coordinates.` };
  }
  if (!finite(summary.inputs.valid_fraction) || summary.inputs.valid_fraction < 0 || summary.inputs.valid_fraction > 1) return { ok: false, reason: "The input valid fraction is invalid." };
  if (targets != null && (!nonnegativeInt(targets.n_cells) || !nonnegativeInt(targets.n_valid) || targets.n_valid > targets.n_cells || !finite(targets.valid_fraction) || targets.valid_fraction < 0 || targets.valid_fraction > 1)) return { ok: false, reason: "Target data-quality counts are invalid." };

  const invalidChannels = [];
  const channelRows = Object.entries(channels).map(([name, item]) => {
    const valid = item && nonnegativeInt(item.n_cells) && nonnegativeInt(item.n_valid) && nonnegativeInt(item.n_missing) && item.n_valid + item.n_missing === item.n_cells && (item.valid_fraction == null || (finite(item.valid_fraction) && item.valid_fraction >= 0 && item.valid_fraction <= 1));
    if (!valid) invalidChannels.push(name);
    return {
      name,
      description: cleanText(item?.description),
      units: cleanText(raw.provenance?.variables?.[name]?.units),
      n_cells: valid ? item.n_cells : null,
      n_valid: valid ? item.n_valid : null,
      n_missing: valid ? item.n_missing : null,
      valid_fraction: finite(item?.valid_fraction) && item.valid_fraction >= 0 && item.valid_fraction <= 1 ? item.valid_fraction : null,
      invalid: !valid,
    };
  });
  const availableDateMetadata = {
    count: dates?.length ?? raw.dates?.count ?? null,
    start: dates?.[0] ?? raw.dates?.min_date ?? null,
    end: dates?.at(-1) ?? raw.dates?.max_date ?? null,
    dates,
  };
  const depthsFromTargets = Object.keys(targets?.per_depth ?? {}).map(Number).filter(finite).sort((a, b) => a - b);
  const modelDepths = Array.isArray(modelPayload?.architecture?.depths)
    ? modelPayload.architecture.depths.filter((depth) => finite(depth) && depth >= 0).sort((a, b) => a - b)
    : [];
  const targetVariables = Array.isArray(targets?.variables) ? targets.variables.filter((value) => typeof value === "string") : [];
  const invalidStats = [];
  const variablesRaw = raw.provenance?.variables && typeof raw.provenance.variables === "object" ? raw.provenance.variables : {};
  const variables = Object.fromEntries(Object.entries(variablesRaw).map(([name, item]) => {
    const fraction = item?.missing_fraction;
    const validFraction = fraction == null || (finite(fraction) && fraction >= 0 && fraction <= 1);
    const nMissing = item?.n_missing;
    const validMissing = nMissing == null || nonnegativeInt(nMissing);
    if (!validFraction || !validMissing) invalidStats.push(`source variable ${name}`);
    return [name, {
      units: cleanText(item?.units),
      description: cleanText(item?.description),
      dims: Array.isArray(item?.dims) ? item.dims.filter((value) => typeof value === "string") : null,
      shape: Array.isArray(item?.shape) && item.shape.every(nonnegativeInt) ? item.shape : null,
      missing_fraction: validFraction && finite(fraction) ? fraction : null,
      n_missing: validMissing && nonnegativeInt(nMissing) ? nMissing : null,
    }];
  }));
  const checkCount = (value, name) => {
    if (value == null) return null;
    if (nonnegativeInt(value)) return value;
    invalidStats.push(name);
    return null;
  };
  const targetByDepth = {};
  for (const depth of depthsFromTargets) {
    const depthKey = Object.keys(targets.per_depth).find((key) => Number(key) === depth);
    const row = depthKey === undefined ? null : targets.per_depth[depthKey];
    const nCells = checkCount(row?.n_cells, `target ${depth}m total`);
    const nValid = checkCount(row?.n_valid, `target ${depth}m valid`);
    targetByDepth[depth] = nCells !== null && nValid !== null && nValid <= nCells
      ? { n_cells: nCells, n_valid: nValid, n_missing: nCells - nValid, valid_fraction: finite(row?.valid_fraction) && row.valid_fraction >= 0 && row.valid_fraction <= 1 ? row.valid_fraction : null }
      : null;
  }
  let qualityStatus = "N/A";
  if (observed.n_valid === 0) qualityStatus = "UNAVAILABLE";
  else if (observed.n_missing > 0) qualityStatus = "PARTIAL";
  else qualityStatus = "AVAILABLE";

  return { ok: true, value: {
    ...raw,
    dates: availableDateMetadata,
    model_depths: modelDepths,
    target_depths: depthsFromTargets,
    target_variables: targetVariables,
    target_by_depth: targetByDepth,
    source_variables: variables,
    channel_rows: channelRows,
    invalid_channels: invalidChannels,
    invalid_stats: invalidStats,
    quality_status: qualityStatus,
    resolution: raw.grid_metadata ?? null,
    temporal_frequency: cleanText(raw.provenance?.temporal_frequency),
    source_name: cleanText(raw.provenance?.source),
    source_format: cleanText(raw.provenance?.dataset_format),
    preprocessing_version: cleanText(raw.provenance?.version),
    data_period: spatial?.by_period ?? {},
    dateCountDiscrepancy: dates != null && nonnegativeInt(raw.dates?.count) && raw.dates.count !== dates.length,
    dateWarnings: Array.isArray(datesPayload?.warnings) ? datesPayload.warnings : [],
  } };
}
