"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, CalendarDays, Database, Map, RefreshCw, ShieldCheck, TriangleAlert } from "lucide-react";
import { ApiError, dataQuality, dates as fetchDates, modelInfo } from "@/lib/api";
import { validateDataQuality } from "@/lib/dataQualityResponse";

const panel = "rounded-2xl border border-border-subtle bg-surface-900/70 p-4 shadow-panel sm:p-5";
const fmt = (value) => Number.isFinite(value) ? new Intl.NumberFormat().format(value) : "N/A";
const pct = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "N/A";
const display = (value) => value == null || value === "" ? "N/A" : String(value);

function Stat({ label, value, detail }) {
  return <div className="rounded-xl border border-border-subtle bg-surface-950/60 p-3"><p className="text-[10px] font-semibold uppercase tracking-wider text-text-muted">{label}</p><p className="mt-1 break-words font-mono text-sm text-text-primary">{value}</p>{detail ? <p className="mt-1 text-[10px] text-text-muted">{detail}</p> : null}</div>;
}

function CellCoverageMap({ grid }) {
  const canvasRef = useRef(null);
  const [hover, setHover] = useState(null);
  const latEdges = useMemo(() => axisEdges(grid.lat), [grid.lat]);
  const lonEdges = useMemo(() => axisEdges(grid.lon), [grid.lon]);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const context = canvas.getContext("2d");
    if (!context) return undefined;
    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(1, rect.width); const height = 320;
      canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);
      const minLon = Math.min(...lonEdges); const maxLon = Math.max(...lonEdges);
      const minLat = Math.min(...latEdges); const maxLat = Math.max(...latEdges);
      if (!(maxLon > minLon) || !(maxLat > minLat)) return;
      grid.lat.forEach((latitude, row) => grid.lon.forEach((longitude, col) => {
        const value = grid.cells[row][col];
        const lon0 = (Math.min(lonEdges[col], lonEdges[col + 1]) - minLon) / (maxLon - minLon) * width;
        const lon1 = (Math.max(lonEdges[col], lonEdges[col + 1]) - minLon) / (maxLon - minLon) * width;
        const lat0 = (maxLat - Math.max(latEdges[row], latEdges[row + 1])) / (maxLat - minLat) * height;
        const lat1 = (maxLat - Math.min(latEdges[row], latEdges[row + 1])) / (maxLat - minLat) * height;
        context.fillStyle = value === 1 ? "#22d3ee" : value === 0 ? "#f97363" : "#475569";
        context.fillRect(lon0, lat0, Math.max(1, lon1 - lon0 + 0.25), Math.max(1, lat1 - lat0 + 0.25));
      }));
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [grid, latEdges, lonEdges]);

  const onPointer = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const longitude = Math.min(...lonEdges) + (event.clientX - rect.left) / rect.width * (Math.max(...lonEdges) - Math.min(...lonEdges));
    const latitude = Math.max(...latEdges) - (event.clientY - rect.top) / rect.height * (Math.max(...latEdges) - Math.min(...latEdges));
    const row = nearest(grid.lat, latitude); const col = nearest(grid.lon, longitude);
    const value = grid.cells[row]?.[col];
    setHover(value == null ? null : { lat: grid.lat[row], lon: grid.lon[col], value });
  };
  return <div>
    <canvas ref={canvasRef} className="h-80 w-full rounded-lg border border-border-subtle bg-surface-950" role="img" aria-label="Observed input coverage grid, from actual tensor masks" onPointerMove={onPointer} onPointerLeave={() => setHover(null)} />
    <div className="mt-2 flex flex-wrap gap-4 text-caption text-text-secondary"><span><i className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm bg-cyan-400" />Valid observed cell</span><span><i className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm bg-[#f97363]" />No valid physical input observed</span><span><i className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm bg-slate-600" />Outside ocean mask</span></div>
    <p className="mt-2 min-h-5 text-caption text-text-muted">{hover ? `${hover.lat}° latitude, ${hover.lon}° longitude · ${hover.value === 1 ? "Valid observed input" : hover.value === 0 ? "No valid input observed" : "Outside ocean mask"}` : grid.scope}</p>
  </div>;
}

function axisEdges(axis) {
  if (axis.length === 1) return [axis[0] - 0.5, axis[0] + 0.5];
  const edges = [axis[0] - (axis[1] - axis[0]) / 2];
  for (let index = 1; index < axis.length; index += 1) edges.push((axis[index - 1] + axis[index]) / 2);
  edges.push(axis.at(-1) + (axis.at(-1) - axis.at(-2)) / 2);
  return edges;
}
function nearest(axis, value) { let found = 0; for (let i = 1; i < axis.length; i += 1) if (Math.abs(axis[i] - value) < Math.abs(axis[found] - value)) found = i; return found; }

function TemporalTimeline({ dates }) {
  const canvasRef = useRef(null);
  const [hover, setHover] = useState(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !dates?.length) return undefined;
    const draw = () => {
      const rect = canvas.getBoundingClientRect(); const width = Math.max(rect.width, 1); const height = 88; const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
      const context = canvas.getContext("2d"); if (!context) return;
      context.setTransform(ratio, 0, 0, ratio, 0, 0); context.clearRect(0, 0, width, height);
      const first = Date.parse(`${dates[0]}T00:00:00Z`); const last = Date.parse(`${dates.at(-1)}T00:00:00Z`);
      dates.forEach((date) => { const stamp = Date.parse(`${date}T00:00:00Z`); const x = first === last ? width / 2 : 8 + (stamp - first) / (last - first) * (width - 16); context.fillStyle = "#22d3ee"; context.beginPath(); context.arc(x, height / 2, 2.5, 0, Math.PI * 2); context.fill(); });
    };
    draw(); const observer = new ResizeObserver(draw); observer.observe(canvas); return () => observer.disconnect();
  }, [dates]);
  const handleMove = (event) => {
    const rect = event.currentTarget.getBoundingClientRect(); const first = Date.parse(`${dates[0]}T00:00:00Z`); const last = Date.parse(`${dates.at(-1)}T00:00:00Z`);
    const stamp = first + Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * (last - first);
    let bestDate = dates[0]; let bestGap = Infinity;
    dates.forEach((date) => { const gap = Math.abs(Date.parse(`${date}T00:00:00Z`) - stamp); if (gap < bestGap) { bestGap = gap; bestDate = date; } });
    setHover(bestDate);
  };
  if (!dates) return <div className="grid h-24 place-items-center text-small text-text-muted">Available-date list unavailable.</div>;
  if (!dates.length) return <div className="grid h-24 place-items-center text-small text-text-muted">No available dates reported.</div>;
  return <div><canvas ref={canvasRef} role="img" aria-label={`Timeline showing ${dates.length} backend-reported available dates`} className="h-[88px] w-full" onPointerMove={handleMove} onPointerLeave={() => setHover(null)} /><div className="flex justify-between text-caption text-text-muted"><span>{dates[0]}</span><span>{hover ? `Available: ${hover}` : `${dates.length} actual dates · no dates interpolated`}</span><span>{dates.at(-1)}</span></div></div>;
}

function AvailabilityBar({ fraction }) {
  if (!Number.isFinite(fraction)) return <span className="text-caption text-text-muted">N/A</span>;
  return <div className="h-2.5 w-full overflow-hidden rounded-full bg-surface-950"><div className="h-full rounded-full bg-gradient-to-r from-cyan-600 to-teal-300" style={{ width: `${Math.max(0, Math.min(1, fraction)) * 100}%` }} /></div>;
}

function CalendarList({ dates }) {
  const [page, setPage] = useState(0); const pageSize = 60;
  useEffect(() => setPage(0), [dates]);
  if (dates == null) return <p className="text-small text-text-muted">Available-date list unavailable.</p>;
  if (!dates.length) return <p className="text-small text-text-muted">No dates are available in the loaded dataset.</p>;
  const count = Math.ceil(dates.length / pageSize); const current = Math.min(page, count - 1);
  return <><div className="flex flex-wrap gap-2">{dates.slice(current * pageSize, (current + 1) * pageSize).map((date) => <span key={date} className="rounded-md border border-border-subtle bg-surface-950 px-2 py-1 font-mono text-[11px] text-text-secondary">{date}</span>)}</div>{count > 1 ? <div className="mt-3 flex items-center gap-3"><button disabled={current === 0} onClick={() => setPage(current - 1)} className="rounded border border-border-subtle px-2 py-1 text-caption disabled:opacity-40">Previous</button><span className="text-caption text-text-muted">Page {current + 1} of {count}</span><button disabled={current + 1 === count} onClick={() => setPage(current + 1)} className="rounded border border-border-subtle px-2 py-1 text-caption disabled:opacity-40">Next</button></div> : null}</>;
}

export default function DataQualityClient() {
  const [state, setState] = useState({ loading: true, error: null, data: null, datesError: null });
  const requestId = useRef(0); const controllerRef = useRef(null);
  const load = useCallback(async () => {
    const id = ++requestId.current; controllerRef.current?.abort(); const controller = new AbortController(); controllerRef.current = controller;
    setState({ loading: true, error: null, data: null, datesError: null });
    const [qualityResult, datesResult, modelResult] = await Promise.allSettled([
      dataQuality({}, { signal: controller.signal, timeoutMs: 30000 }),
      fetchDates({ signal: controller.signal, timeoutMs: 30000 }),
      modelInfo({ signal: controller.signal, timeoutMs: 30000 }),
    ]);
    if (id !== requestId.current) return;
    if (qualityResult.status === "rejected") { setState({ loading: false, error: qualityResult.reason?.message || "Data quality unavailable.", data: null, datesError: null }); return; }
    const datePayload = datesResult.status === "fulfilled" ? datesResult.value : null;
    const modelPayload = modelResult.status === "fulfilled" ? modelResult.value : null;
    const validation = validateDataQuality(qualityResult.value, datePayload, modelPayload);
    if (!validation.ok) { setState({ loading: false, error: validation.reason, data: null, datesError: null }); return; }
    setState({ loading: false, error: null, data: validation.value, datesError: datesResult.status === "rejected" ? datesResult.reason?.message || "Available dates could not be loaded." : null });
  }, []);
  useEffect(() => { load(); return () => { requestId.current += 1; controllerRef.current?.abort(); }; }, [load]);
  const [targetDepth, setTargetDepth] = useState(null);
  useEffect(() => { if (state.data?.target_depths?.length && !state.data.target_depths.includes(targetDepth)) setTargetDepth(state.data.target_depths[0]); }, [state.data, targetDepth]);
  const data = state.data;
  const targetStats = data?.target_by_depth?.[targetDepth] ?? null;
  const sourceVariableRows = useMemo(() => {
    if (!data) return [];
    const source = data.source_variables ?? {};
    const rows = [];
    const names = new Set([...Object.keys(source), ...data.channel_rows.map((row) => row.name), ...data.target_variables]);
    for (const name of names) {
      const channel = data.channel_rows.find((item) => item.name === name);
      const target = data.target_variables.includes(name) ? data.targets : null;
      const metadata = source[name] ?? {};
      rows.push({ name, description: channel?.description ?? metadata.description ?? null, units: metadata.units ?? channel?.units ?? null, availability: channel?.valid_fraction ?? (target ? data.targets?.valid_fraction : null) ?? (Number.isFinite(metadata.missing_fraction) ? 1 - metadata.missing_fraction : null), valid: channel?.n_valid ?? (target ? data.targets?.n_valid : null) ?? null, missing: channel?.n_missing ?? (target ? data.targets?.n_missing : null) ?? metadata.n_missing ?? null, scope: channel ? "processed input mask" : target ? "target mask" : "source metadata" });
    }
    return rows;
  }, [data]);

  return <main className="mx-auto flex w-full max-w-[1500px] flex-col gap-5 p-4 sm:p-6 xl:p-8">
    <header className="flex flex-wrap items-start justify-between gap-4"><div><p className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-accent-300"><ShieldCheck size={14} />Dataset provenance</p><h1 className="mt-1 text-2xl font-semibold text-text-primary sm:text-3xl">Data Quality</h1><p className="mt-2 max-w-3xl text-small text-text-secondary">A transparent view of the loaded tensor dataset, its actual coverage masks, variables, and available dates.</p></div><button onClick={load} disabled={state.loading} className="inline-flex items-center gap-2 rounded-lg border border-border-subtle bg-surface-900 px-3 py-2 text-small text-text-secondary disabled:opacity-50"><RefreshCw size={14} className={state.loading ? "animate-spin" : ""} />Refresh</button></header>
    {state.loading ? <section role="status" className={`${panel} grid min-h-48 place-items-center text-small text-text-muted`}><span><Activity size={16} className="mr-2 inline animate-pulse text-cyan-300" />Loading dataset quality metadata…</span></section> : state.error ? <section role="alert" className={`${panel} flex flex-wrap items-center justify-between gap-4`}><div><h2 className="font-semibold text-text-primary">Data quality unavailable</h2><p className="mt-1 text-small text-text-secondary">{state.error}</p></div><button onClick={load} className="inline-flex items-center gap-2 rounded-md border border-border-subtle px-3 py-2 text-small"><RefreshCw size={14} />Retry</button></section> : data ? <>
      {data.is_synthetic ? <div role="status" className="rounded-xl border-2 border-amber-400/50 bg-amber-400/5 p-3 text-small text-amber-100"><TriangleAlert size={15} className="mr-2 inline" />SYNTHETIC / DEMO DATA — {data.disclaimer || data.data_mode}. These metadata and quality statistics describe a synthetic dataset, not observational coverage.</div> : <div className="flex flex-wrap items-center gap-2"><span className="rounded-full border border-cyan-300/30 bg-cyan-300/5 px-2.5 py-1 text-[10px] font-semibold tracking-wider text-cyan-200">{data.data_mode}</span><span className="text-caption text-text-muted">Loaded tensor dataset</span></div>}
      <section className={panel}><div className="mb-4 flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-base font-semibold text-text-primary">DATASET METADATA</h2><p className="mt-1 text-caption text-text-muted">Metadata unavailable in the source is shown as N/A.</p></div><span className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold tracking-wider ${data.quality_status === "AVAILABLE" ? "border-emerald-300/30 text-emerald-200" : data.quality_status === "PARTIAL" ? "border-amber-300/30 text-amber-200" : "border-slate-400/30 text-slate-300"}`}>INPUT COVERAGE: {data.quality_status}</span></div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4"><Stat label="Data source" value={data.source_name} detail={data.source_format ? `Format: ${data.source_format}` : "Format: N/A"} /><Stat label="Dataset identifier" value={data.provenance?.dataset_identifier} detail={`Mode: ${data.data_mode}`} /><Stat label="Time coverage" value={data.dates.start && data.dates.end ? `${data.dates.start} → ${data.dates.end}` : "N/A"} detail={`Available dates: ${fmt(data.dates.count)}`} /><Stat label="Temporal frequency" value={data.temporal_frequency} detail="From preprocessing metadata" /><Stat label="Latitude coverage" value={data.spatial_coverage.lat_range?.map((n) => `${n}°`).join(" → ") ?? "N/A"} detail={`${fmt(data.grid_metadata?.lat_points)} coordinate points`} /><Stat label="Longitude coverage" value={data.spatial_coverage.lon_range?.map((n) => `${n}°`).join(" → ") ?? "N/A"} detail={`${fmt(data.grid_metadata?.lon_points)} coordinate points`} /><Stat label="Latitude resolution" value={data.grid_metadata?.lat_resolution?.uniform ? `${data.grid_metadata.lat_resolution.degrees}°` : data.grid_metadata?.lat_resolution?.uniform === false ? "Irregular" : "N/A"} detail="Measured from bundle coordinates" /><Stat label="Longitude resolution" value={data.grid_metadata?.lon_resolution?.uniform ? `${data.grid_metadata.lon_resolution.degrees}°` : data.grid_metadata?.lon_resolution?.uniform === false ? "Irregular" : "N/A"} detail="Measured from bundle coordinates" /><Stat label="Grid shape" value={data.spatial_coverage.grid_shape?.join(" × ") ?? "N/A"} detail="Latitude × longitude" /><Stat label="Input variables" value={fmt(data.channel_rows.length)} detail="Channels reported by backend" /><Stat label="Target variables" value={data.target_variables.length ? data.target_variables.join(", ") : "N/A"} detail="Targets present in loaded tensors" /><Stat label="Valid physical input cells" value={fmt(data.observed_input_cells.n_valid)} detail={data.observed_input_cells.scope} /></div>
        <div className="mt-4 grid gap-3 sm:grid-cols-2"><Stat label="Source tensor bundle" value={data.source_tensors_path} /><Stat label="Preprocessing record" value={data.preprocessing_version ? `v${data.preprocessing_version}` : "N/A"} detail={`${data.provenance?.preprocessing_pipeline || "Pipeline N/A"} · created ${data.provenance?.preprocessing_created_at || "N/A"}`} /><Stat label="Dataset version / provider" value={data.provenance?.dataset_identifier || "N/A"} detail="No dataset identifier or provider supplied" /></div>
      </section>

      <div className="grid gap-5 xl:grid-cols-2"><section className={panel}><div className="mb-3 flex items-center gap-2"><Map size={16} className="text-cyan-300" /><div><h2 className="text-base font-semibold text-text-primary">SPATIAL COVERAGE</h2><p className="text-caption text-text-muted">Any physical input variable observed on at least one available date</p></div></div><CellCoverageMap grid={data.coverage_grid} /><div className="mt-3 grid grid-cols-3 gap-2"><Stat label="Valid grid cells" value={fmt(data.coverage_grid.valid_cells)} /><Stat label="Missing grid cells" value={fmt(data.coverage_grid.missing_cells)} /><Stat label="Outside ocean mask" value={data.coverage_grid.outside_cells == null ? "N/A" : fmt(data.coverage_grid.outside_cells)} /></div></section>
        <section className={panel}><div className="mb-3 flex items-center gap-2"><Activity size={16} className="text-cyan-300" /><div><h2 className="text-base font-semibold text-text-primary">MISSING DATA</h2><p className="text-caption text-text-muted">Physical input masks across available dates and the full grid</p></div></div><div className="grid gap-3 sm:grid-cols-2"><Stat label="Valid values" value={fmt(data.observed_input_cells.n_valid)} detail={pct(data.observed_input_cells.n_cells ? data.observed_input_cells.n_valid / data.observed_input_cells.n_cells : NaN)} /><Stat label="Missing values" value={fmt(data.observed_input_cells.n_missing)} detail={pct(data.observed_input_cells.n_cells ? data.observed_input_cells.n_missing / data.observed_input_cells.n_cells : NaN)} /><Stat label="Total scope" value={fmt(data.observed_input_cells.n_cells)} detail={data.observed_input_cells.scope} /><Stat label="All model input cells" value={`${fmt(data.input_cells.n_valid)} valid / ${fmt(data.input_cells.n_cells)} total`} detail={data.input_cells.scope} /></div><div className="mt-4"><div className="mb-1 flex justify-between text-caption text-text-muted"><span>Valid</span><span>Missing</span></div><div className="flex h-4 overflow-hidden rounded-full bg-surface-950" role="img" aria-label={`Physical input cells: ${data.observed_input_cells.n_valid} valid and ${data.observed_input_cells.n_missing} missing`}>{data.observed_input_cells.n_cells ? <><div className="bg-cyan-400" style={{ width: `${data.observed_input_cells.n_valid / data.observed_input_cells.n_cells * 100}%` }} /><div className="bg-rose-400" style={{ width: `${data.observed_input_cells.n_missing / data.observed_input_cells.n_cells * 100}%` }} /></> : null}</div></div>{targetStats ? <div className="mt-4 rounded-lg border border-border-subtle p-3"><label className="flex flex-wrap items-center justify-between gap-3 text-caption text-text-muted">Target-data depth<select value={targetDepth ?? ""} onChange={(event) => setTargetDepth(Number(event.target.value))} className="rounded-md border border-border-subtle bg-surface-950 px-2 py-1 text-small text-text-primary">{data.target_depths.map((depth) => <option key={depth} value={depth}>{depth} m</option>)}</select></label><p className="mt-2 text-small text-text-secondary">Target mask at {targetDepth} m: {fmt(targetStats.n_valid)} valid / {fmt(targetStats.n_cells)} total · {fmt(targetStats.n_missing)} missing · valid fraction {pct(targetStats.valid_fraction)}</p><p className="mt-1 text-[10px] text-text-muted">Counts apply to this target depth over all dates and grid cells. They are separate from input-variable counts.</p></div> : data.targets == null ? <p className="mt-4 text-caption text-text-muted">Target missingness: N/A; the backend reports no target tensor.</p> : null}</section></div>

      <section className={panel}><div className="mb-3 flex items-center gap-2"><Database size={16} className="text-cyan-300" /><div><h2 className="text-base font-semibold text-text-primary">VARIABLE AVAILABILITY</h2><p className="text-caption text-text-muted">Availability reflects actual source metadata and backend masks; no units are assumed.</p></div></div><div className="overflow-x-auto"><table className="w-full min-w-[650px] text-left text-small"><thead className="border-b border-border-subtle text-caption text-text-muted"><tr><th className="p-2">Variable</th><th className="p-2">Description</th><th className="p-2">Units</th><th className="p-2">Availability</th><th className="p-2">Valid / missing</th><th className="p-2">Scope</th></tr></thead><tbody>{sourceVariableRows.map((row) => <tr key={row.name} className="border-b border-border-subtle/60"><th className="p-2 font-mono font-medium text-text-primary">{row.name}</th><td className="p-2 text-text-secondary">{row.description ?? "N/A"}</td><td className="p-2 text-text-secondary">{row.units ?? "N/A"}</td><td className="min-w-32 p-2"><div className="flex items-center gap-2"><AvailabilityBar fraction={row.availability} /><span className="w-14 text-right text-caption text-text-secondary">{pct(row.availability)}</span></div></td><td className="p-2 font-mono text-caption text-text-secondary">{row.valid == null ? "N/A" : `${fmt(row.valid)} / ${row.missing == null ? "N/A" : fmt(row.missing)}`}</td><td className="p-2 text-caption text-text-muted">{row.scope}</td></tr>)}</tbody></table></div>{!sourceVariableRows.length ? <p className="mt-3 text-small text-text-muted">Variable metadata unavailable.</p> : null}<p className="mt-3 text-caption text-text-muted">Variable × depth availability: not provided by backend.</p></section>

      <div className="grid gap-5 xl:grid-cols-2"><section className={panel}><div className="mb-3 flex items-center gap-2"><CalendarDays size={16} className="text-cyan-300" /><div><h2 className="text-base font-semibold text-text-primary">TEMPORAL COVERAGE</h2><p className="text-caption text-text-muted">Each marker is a date returned by the backend; spacing follows actual timestamps.</p></div></div>{state.datesError ? <p role="status" className="mb-2 text-caption text-amber-200">The separate available-dates endpoint failed: {state.datesError}</p> : null}{data.dateCountDiscrepancy || data.dateWarnings.length ? <p role="status" className="mb-2 text-caption text-amber-200">Date metadata differs from the unique dates returned. The chart and list show only those actual returned dates.</p> : null}<TemporalTimeline dates={data.dates.dates} /><div className="mt-4 grid grid-cols-3 gap-2"><Stat label="Start date" value={display(data.dates.start)} /><Stat label="End date" value={display(data.dates.end)} /><Stat label="Available dates" value={fmt(data.dates.count)} /></div><details className="mt-4 rounded-lg border border-border-subtle p-3"><summary className="cursor-pointer text-small text-text-secondary">Browse actual available dates</summary><p className="my-3 text-caption text-text-muted">Only dates returned by /dates are listed. No missing dates are generated.</p><CalendarList dates={data.dates.dates} /></details></section>
        <section className={panel}><div className="mb-3"><h2 className="text-base font-semibold text-text-primary">DEPTH LEVELS</h2><p className="text-caption text-text-muted">Depths come from the loaded target bundle and model metadata.</p></div><p className="mb-2 text-caption text-text-muted">Dataset target depths</p><div className="flex flex-wrap gap-2">{data.target_depths.length ? data.target_depths.map((depth) => <span key={depth} className="rounded-full border border-cyan-300/25 bg-cyan-300/5 px-2.5 py-1 text-caption text-cyan-100">{depth} m</span>) : <span className="text-small text-text-muted">N/A</span>}</div><p className="mb-2 mt-4 text-caption text-text-muted">Model depths</p><div className="flex flex-wrap gap-2">{data.model_depths.length ? data.model_depths.map((depth) => <span key={depth} className="rounded-full border border-border-subtle px-2.5 py-1 text-caption text-text-secondary">{depth} m</span>) : <span className="text-small text-text-muted">N/A (model metadata unavailable)</span>}</div><div className="mt-4 grid gap-2 sm:grid-cols-2"><Stat label="Target cell scope" value={data.targets ? fmt(data.targets.n_cells) : "N/A"} detail={data.targets ? `${fmt(data.targets.n_valid)} valid · ${fmt(data.targets.n_missing ?? (data.targets.n_cells - data.targets.n_valid))} missing across all target depths` : "No target tensor reported"} /><Stat label="Dataset scope" value="Loaded tensor bundle" detail="No other dataset products are exposed by this API." /></div></section></div>
    </> : null}
  </main>;
}
