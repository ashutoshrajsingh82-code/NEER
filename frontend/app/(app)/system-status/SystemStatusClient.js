"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Download, RefreshCw } from "lucide-react";
import { dates, health, modelInfo, netcdfExport } from "@/lib/api";
import { useNeerContext } from "../dashboard/_context/NeerContext";

const NA = "N/A";
const numberText = (value) => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value)) ? Number(value).toLocaleString() : NA;
const pretty = (value) => value == null || value === "" ? NA : String(value);

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function csvCell(value) {
  if (value === null || value === undefined || (typeof value === "number" && !Number.isFinite(value))) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function StatusCard({ label, value, detail }) {
  return <section className="rounded-xl border border-border-subtle bg-surface-900 p-4">
    <h2 className="text-[11px] font-semibold uppercase tracking-widest text-text-muted">{label}</h2>
    <p className="mt-2 break-words font-mono text-base text-text-primary">{value}</p>
    {detail ? <p className="mt-1 break-words text-xs text-text-muted">{detail}</p> : null}
  </section>;
}

export default function SystemStatusClient() {
  const { selectedPoint, date, depth, dataMode, pointInspection, variableMode } = useNeerContext();
  const [status, setStatus] = useState({ health: null, model: null, dates: null, error: null, lastChecked: null, requestMs: null });
  const [refreshing, setRefreshing] = useState(false);
  const [bounds, setBounds] = useState({ latMin: "", latMax: "", lonMin: "", lonMax: "" });
  const [exportError, setExportError] = useState("");
  const busy = useRef(false);
  const activeController = useRef(null);

  const refresh = useCallback(async () => {
    if (busy.current) return;
    busy.current = true;
    const controller = new AbortController();
    activeController.current = controller;
    setRefreshing(true);
    let elapsed = null;
    let healthCheckedAt = null;
    const healthRequest = (async () => {
      const started = performance.now();
      try {
        const result = await health({ signal: controller.signal });
        healthCheckedAt = new Date().toISOString();
        return result;
      } finally { elapsed = performance.now() - started; }
    })();
    const results = await Promise.allSettled([healthRequest, modelInfo({ signal: controller.signal }), dates({ signal: controller.signal })]);
    if (activeController.current !== controller) return;
    setStatus((previous) => {
      const next = { ...previous, requestMs: elapsed, error: null };
      if (results[0].status === "fulfilled") {
        next.health = results[0].value;
        next.lastChecked = healthCheckedAt;
      } else {
        next.health = null;
        next.error = results[0].reason?.message || "Health request failed";
      }
      next.model = results[1].status === "fulfilled" ? results[1].value : null;
      next.dates = results[2].status === "fulfilled" ? results[2].value : null;
      return next;
    });
    busy.current = false;
    if (activeController.current === controller) activeController.current = null;
    setRefreshing(false);
  }, []);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 60000);
    return () => {
      window.clearInterval(timer);
      activeController.current?.abort();
      activeController.current = null;
      busy.current = false;
    };
  }, [refresh]);

  const components = status.health?.components || {};
  const apiStatus = !status.health ? (status.error ? "OFFLINE" : "CHECKING") :
    status.health.status === "ok" && components.api?.status === "ok" ? "ONLINE" : "DEGRADED";
  const modelLoaded = components.model?.status === "ok";
  const checkpoint = status.model?.checkpoint || {};
  const modelRuntime = status.model?.runtime || {};
  const dataComponent = components.data || {};
  const availableDates = status.dates?.dates || [];
  const depths = status.model?.architecture?.depths || [];
  const reconstruction = pointInspection?.data;
  const matchingResult = reconstruction && reconstruction.date === date && reconstruction.lat === selectedPoint?.lat && reconstruction.lon === selectedPoint?.lon && (depth == null || Math.abs(Number(reconstruction.depth) - Number(depth)) < 1e-6);

  const exportPoint = (format) => {
    setExportError("");
    if (!matchingResult) return;
    const payload = {
      metadata: { date: reconstruction.date, latitude: reconstruction.lat, longitude: reconstruction.lon, depth: reconstruction.depth, data_mode: reconstruction.data_mode, source: "NEER point reconstruction API" },
      data: reconstruction,
    };
    const base = `neer_${reconstruction.date}_${reconstruction.depth}m_point`;
    if (format === "json") {
      saveBlob(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }), `${base}.json`);
      return;
    }
    const rows = [{ date: reconstruction.date, latitude: reconstruction.lat, longitude: reconstruction.lon, depth: reconstruction.depth, variable: "temperature", value: reconstruction.temperature }];
    if (reconstruction.anomaly !== undefined) rows.push({ date: reconstruction.date, latitude: reconstruction.lat, longitude: reconstruction.lon, depth: reconstruction.depth, variable: "anomaly", value: reconstruction.anomaly });
    if (reconstruction.climatology !== undefined && reconstruction.climatology !== null) rows.push({ date: reconstruction.date, latitude: reconstruction.lat, longitude: reconstruction.lon, depth: reconstruction.depth, variable: "climatology", value: reconstruction.climatology });
    const headers = ["date", "latitude", "longitude", "depth", "variable", "value"];
    const csv = [headers.join(","), ...rows.map((row) => headers.map((header) => csvCell(row[header])).join(","))].join("\r\n");
    saveBlob(new Blob([csv], { type: "text/csv;charset=utf-8" }), `${base}.csv`);
  };

  const exportNetcdf = async () => {
    setExportError("");
    const values = Object.fromEntries(Object.entries(bounds).map(([key, value]) => [key, Number(value)]));
    if (!date || depth == null || Object.values(bounds).some((value) => value === "") || !Object.values(values).every(Number.isFinite) || values.latMin >= values.latMax || values.lonMin >= values.lonMax) {
      setExportError("Choose a current date and depth, then enter a valid non-empty latitude/longitude box.");
      return;
    }
    try {
      const result = await netcdfExport({ ...values, date, depth });
      saveBlob(result.blob, result.filename || `neer_reconstruction_${date}.nc`);
    } catch (error) {
      setExportError(error?.message || "NetCDF export failed. No file was offered for download.");
    }
  };

  const setBound = (key, value) => setBounds((current) => ({ ...current, [key]: value }));
  const inputClass = "w-full rounded-md border border-border-subtle bg-surface-950 px-2 py-2 font-mono text-sm text-text-primary";

  return <main className="mx-auto w-full max-w-7xl space-y-6 px-4 py-6 md:px-8">
    <header className="flex flex-wrap items-end justify-between gap-3">
      <div><p className="text-[11px] font-semibold uppercase tracking-[.2em] text-accent-300">Operations</p><h1 className="mt-1 text-2xl font-semibold text-text-primary">System Status</h1><p className="mt-1 text-sm text-text-secondary">Live backend, model, and dataset information.</p></div>
      <div className="flex items-center gap-3 text-xs text-text-muted"><span>Last checked: {status.lastChecked ? new Date(status.lastChecked).toLocaleString() : NA}</span><button type="button" onClick={refresh} disabled={refreshing} className="inline-flex items-center gap-2 rounded-lg border border-border-subtle px-3 py-2 text-text-primary disabled:opacity-50"><RefreshCw size={14} className={refreshing ? "animate-spin" : ""} />Refresh</button></div>
    </header>
    {status.error ? <p role="alert" className="rounded-lg border border-rose-400/30 bg-rose-400/5 p-3 text-sm text-rose-200">Health request unavailable: {status.error}</p> : null}
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      <StatusCard label="API status" value={apiStatus} detail={status.requestMs == null ? "Response: N/A" : `Health response: ${numberText(status.requestMs)} ms (measured request round trip)`} />
      <StatusCard label="Model status" value={modelLoaded ? "LOADED" : components.model?.status === "unavailable" ? "NOT LOADED" : "N/A"} detail={components.model?.detail} />
      <StatusCard label="Checkpoint" value={pretty(checkpoint.filename || checkpoint.path?.split(/[\\/]/).pop())} detail={checkpoint.epoch == null ? undefined : `Epoch ${checkpoint.epoch}`} />
      <StatusCard label="Device" value={pretty(modelRuntime.device || components.model?.device)} />
      <StatusCard label="PyTorch" value={pretty(status.model?.pytorch_version)} />
      <StatusCard label="Model version" value={pretty(checkpoint.model_version || checkpoint.version)} detail={status.model?.architecture ? `Architecture: ${status.model.architecture.embed_dim == null ? NA : `embedding ${status.model.architecture.embed_dim}`}` : undefined} />
      <StatusCard label="Dataset" value={dataComponent.status === "ok" ? "AVAILABLE" : "N/A"} detail={dataComponent.grid_shape ? `Grid ${dataComponent.grid_shape.join(" × ")}; ${numberText(dataComponent.n_timesteps)} time steps` : NA} />
      <StatusCard label="Data mode" value={pretty(dataComponent.data_mode || status.dates?.dataMode)} detail={dataComponent.is_synthetic === true ? "Synthetic data reported by backend" : undefined} />
      <StatusCard label="Inference latency" value={matchingResult ? `${numberText(reconstruction.latency_ms)} ms` : NA} detail={matchingResult ? (reconstruction.cache_hit ? "Backend reconstruction latency · cached result" : "Backend reconstruction latency · model request") : "Select a map point and wait for its current reconstruction"} />
      <StatusCard label="Available dates" value={status.dates ? numberText(availableDates.length) : NA} detail={status.dates ? `${availableDates[0] || NA} to ${availableDates.at(-1) || NA}` : "Dates unavailable"} />
      <StatusCard label="Available depths" value={depths.length ? `${depths.length} levels` : NA} detail={depths.length ? `${depths.join(", ")} m` : "Model depth list unavailable"} />
    </div>
    <section className="rounded-xl border border-border-subtle bg-surface-900 p-5">
      <div><h2 className="font-semibold text-text-primary">Scientific export</h2><p className="mt-1 text-sm text-text-secondary">JSON and CSV export the selected point’s completed reconstruction. NetCDF requests a backend reconstruction over the box below and is downloaded only after backend validation.</p></div>
      <div className="mt-4 rounded-lg bg-surface-950 p-3 text-sm text-text-secondary"><strong className="text-text-primary">Export context</strong><p className="mt-1">Date: {pretty(date)} · Depth: {depth == null ? NA : `${depth} m`} · Point: {selectedPoint ? `${selectedPoint.lat}, ${selectedPoint.lon}` : "not selected"} · Field: {variableMode === "anomaly" ? "ANOMALY" : "MODEL OUTPUT"} · Data mode: {pretty(dataMode)}</p><p className="mt-1">Point reconstruction: {matchingResult ? "ready" : pointInspection?.isLoading ? "in progress" : "unavailable or does not match current selection"}</p></div>
      <div className="mt-4 flex flex-wrap gap-2"><button type="button" disabled={!matchingResult} onClick={() => exportPoint("json")} className="inline-flex items-center gap-2 rounded-lg border border-border-subtle px-3 py-2 text-sm text-text-primary disabled:opacity-40"><Download size={14} />JSON</button><button type="button" disabled={!matchingResult} onClick={() => exportPoint("csv")} className="inline-flex items-center gap-2 rounded-lg border border-border-subtle px-3 py-2 text-sm text-text-primary disabled:opacity-40"><Download size={14} />CSV</button></div>
      <div className="mt-5 border-t border-border-subtle pt-4"><h3 className="text-sm font-medium text-text-primary">NetCDF grid bounds</h3><p className="mt-1 text-xs text-text-muted">Coordinates define the exported grid region; date and depth follow the shared selection above.</p><div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-4">{[["latMin", "Latitude min"], ["latMax", "Latitude max"], ["lonMin", "Longitude min"], ["lonMax", "Longitude max"]].map(([key, label]) => <label key={key} className="grid gap-1 text-xs text-text-muted">{label}<input aria-label={label} type="number" step="any" value={bounds[key]} onChange={(event) => setBound(key, event.target.value)} className={inputClass} /></label>)}</div><button type="button" disabled={!date || depth == null} onClick={exportNetcdf} className="mt-3 inline-flex items-center gap-2 rounded-lg bg-accent-500 px-3 py-2 text-sm font-medium text-surface-950 disabled:opacity-40"><Download size={14} />Generate validated NetCDF</button>{exportError ? <p role="alert" className="mt-2 text-sm text-rose-200">{exportError}</p> : null}</div>
    </section>
  </main>;
}
