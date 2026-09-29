"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, MapPin, RefreshCw } from "lucide-react";
import DateControls from "../dashboard/_components/DateControls";
import { useNeerContext } from "../dashboard/_context/NeerContext";
import { ApiError, explainability as fetchExplainability } from "@/lib/api";
import { formatDepth, formatLat, formatLon, isWithinDomain, OCEAN_DOMAIN } from "@/lib/oceanDomain";
import { validateExplainabilityResponse } from "@/lib/explainabilityResponse";
import { isDateInDataRange } from "@/lib/dateDepthModel";

const card = "rounded-xl border border-border-subtle bg-surface-900/70 p-4";
const inputClass = "w-full rounded-lg border border-border-subtle bg-surface-900 px-3 py-2 text-small font-mono text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-300";
const descriptions = {
  sst: "Surface temperature input field",
  sss: "Surface salinity input field",
  sla: "Sea-level anomaly input field",
  u_current: "Eastward ocean-current component",
  v_current: "Northward ocean-current component",
  u_wind: "Eastward wind component",
  v_wind: "Northward wind component",
};
const formatAttribution = (value) => value === null ? "N/A" : (value > 0 ? "+" : "") + value.toPrecision(7);

export default function ExplainabilityClient() {
  const {
    availableDates, selectedDate, datesStatus, datesError, canStepPrev, canStepNext,
    stepDate, selectDate, retryDates, availableDepths, selectedDepth, selectDepth,
    depthsStatus, depthsError, modelInfoData,
    selectedPoint, selectPoint, variableMode,
  } = useNeerContext();
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [requestState, setRequestState] = useState({ status: "idle", key: null, error: "" });
  const [resultState, setResultState] = useState(null);
  const [retryToken, setRetryToken] = useState(0);
  const [selectedFeature, setSelectedFeature] = useState(null);
  const requestId = useRef(0);
  const cache = useRef(new Map());

  useEffect(() => {
    if (!selectedPoint) return;
    setLatitude(String(selectedPoint.lat));
    setLongitude(String(selectedPoint.lon));
  }, [selectedPoint]);

  const lat = latitude.trim() === "" ? null : Number(latitude);
  const lon = longitude.trim() === "" ? null : Number(longitude);
  const locationValid = Number.isFinite(lat) && Number.isFinite(lon) && isWithinDomain(lat, lon);
  const depthOptions = availableDepths?.length ? availableDepths : [];
  const depthValid = Number.isFinite(selectedDepth) && depthOptions.includes(selectedDepth);
  const dateValid = selectedDate && isDateInDataRange(selectedDate, availableDates);
  const requestParams = useMemo(
    () => dateValid && selectedPoint && isWithinDomain(selectedPoint.lat, selectedPoint.lon) && depthValid
      ? { date: selectedDate, lat: selectedPoint.lat, lon: selectedPoint.lon, depth: selectedDepth }
      : null,
    [dateValid, depthValid, selectedDate, selectedDepth, selectedPoint]
  );
  const requestKey = requestParams ? [requestParams.date, requestParams.lat, requestParams.lon, requestParams.depth].join("|") : null;
  const modelVersion = modelInfoData?.version ?? modelInfoData?.model_version ?? modelInfoData?.checkpoint?.version ?? null;

  useEffect(() => {
    if (!requestParams || !requestKey) {
      setRequestState({ status: "idle", key: null, error: "" });
      return undefined;
    }
    const id = ++requestId.current;
    const cacheKey = modelVersion ? [modelVersion, requestKey].join("|") : null;
    if (cacheKey && cache.current.has(cacheKey)) {
      setResultState({ key: requestKey, value: cache.current.get(cacheKey) });
      setRequestState({ status: "success", key: requestKey, error: "" });
      return undefined;
    }
    const controller = new AbortController();
    setRequestState({ status: "loading", key: requestKey, error: "" });
    const timer = setTimeout(() => {
      fetchExplainability(requestParams, { signal: controller.signal, timeoutMs: 120000 })
        .then((raw) => {
          const validation = validateExplainabilityResponse(raw, requestParams);
          if (!validation.ok) throw new ApiError({ code: "invalid_response", message: validation.reason });
          if (id !== requestId.current || controller.signal.aborted) return;
          setResultState({ key: requestKey, value: validation.value });
          if (cacheKey) cache.current.set(cacheKey, validation.value);
          setRequestState({ status: "success", key: requestKey, error: "" });
        })
        .catch((error) => {
          if (id !== requestId.current || controller.signal.aborted || error?.code === "aborted") return;
          setRequestState({ status: "error", key: requestKey, error: error?.message || "Unable to compute model explanation." });
        });
    }, 180);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [requestKey, requestParams, retryToken, modelVersion]);

  const current = resultState?.key === requestKey ? resultState.value : null;
  const currentError = requestState.status === "error" && requestState.key === requestKey ? requestState.error : "";
  const loading = requestState.status === "loading" && requestState.key === requestKey;
  const selected = current?.features.find((feature) => feature.name === selectedFeature) ?? null;
  const backendDepths = depthOptions;
  const domainError = latitude && longitude && !locationValid ? "Enter coordinates inside the NEER domain (5°N–30°N, 45°E–105°E)." : "";
  const canExplain = Boolean(requestParams);
  const infoEpoch = current?.checkpoint_epoch ?? modelInfoData?.checkpoint?.epoch ?? null;

  function syncMapLocation() {
    if (locationValid) selectPoint({ lat, lon });
  }

  return <main className="mx-auto flex max-w-[1500px] flex-col gap-5">
    <header>
      <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-accent-300">NEER · Model interpretation</p>
      <h1 className="text-2xl font-semibold tracking-tight text-text-primary">Model Explainability</h1>
      <p className="mt-2 max-w-3xl text-small leading-6 text-text-secondary">Integrated Gradients attributes the model decoder output for one date, location, and depth to the seven physical input channels.</p>
    </header>

    <section className={card} aria-label="Explanation selection">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <div>
          <DateControls availableDates={availableDates} date={selectedDate} status={datesStatus} canStepPrev={canStepPrev} canStepNext={canStepNext} onStepDate={stepDate} onSelectDate={selectDate} message={datesError?.message} onRetry={retryDates} />
        </div>
        <label className="flex flex-col gap-1.5 text-caption text-text-muted">LATITUDE · 5°N–30°N
          <input aria-label="Latitude" type="number" step="0.25" min={OCEAN_DOMAIN.latMin} max={OCEAN_DOMAIN.latMax} className={inputClass} value={latitude} placeholder="Select on map or enter latitude" onChange={(event) => setLatitude(event.target.value)} />
        </label>
        <label className="flex flex-col gap-1.5 text-caption text-text-muted">LONGITUDE · 45°E–105°E
          <input aria-label="Longitude" type="number" step="0.25" min={OCEAN_DOMAIN.lonMin} max={OCEAN_DOMAIN.lonMax} className={inputClass} value={longitude} placeholder="Select on map or enter longitude" onChange={(event) => setLongitude(event.target.value)} />
        </label>
        <label className="flex flex-col gap-1.5 text-caption text-text-muted">DEPTH
          <select aria-label="Depth" className={inputClass} value={depthValid ? selectedDepth : ""} disabled={depthsStatus !== "success"} onChange={(event) => selectDepth(Number(event.target.value))}>
            <option value="" disabled>{depthsStatus === "loading" ? "Loading model depths…" : "Select depth"}</option>
            {backendDepths.map((depth) => <option key={depth} value={depth}>{formatDepth(depth)}</option>)}
          </select>
        </label>
      </div>
      {domainError ? <p role="alert" className="mt-3 text-caption text-amber-200">{domainError}</p> : null}
      {locationValid && (!selectedPoint || lat !== selectedPoint.lat || lon !== selectedPoint.lon) ? <button type="button" onClick={syncMapLocation} className="mt-3 rounded border border-border-subtle px-2 py-1 text-caption text-text-primary">Use these coordinates as the global location</button> : null}
      <div className="mt-3 flex flex-wrap items-center gap-3 text-caption text-text-muted">
        <span>Global mode: {variableMode === "anomaly" ? "ANOMALY" : "MODEL OUTPUT"}. Integrated Gradients explains the backend decoder anomaly target.</span>
      </div>
      {!selectedPoint ? <p className="mt-3 flex flex-wrap items-center gap-1 text-caption text-text-muted"><MapPin size={13} />No map point selected. Enter a location or <Link className="text-accent-300 underline" href="/dashboard">select one on the ocean map</Link>.</p> : <p className="mt-3 flex flex-wrap items-center gap-2 text-caption text-text-muted"><MapPin size={13} />Shared model location: {formatLat(selectedPoint.lat)}, {formatLon(selectedPoint.lon)}</p>}
    </section>

    <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4" aria-label="Explainability method and selection context">
      <Meta label="EXPLAINABILITY METHOD" value="Integrated Gradients" emphasis />
      <Meta label="TARGET" value="NEER decoder anomaly residual for selected depth" />
      <Meta label="DATE · DEPTH" value={(selectedDate || "N/A") + " · " + (depthValid ? formatDepth(selectedDepth) : "N/A")} />
      <Meta label="MODEL" value={(current?.model_version ?? "N/A") + (infoEpoch !== null ? " · epoch " + infoEpoch : "")} />
    </section>

    {requestState.status === "idle" && !canExplain ? <section className={card}><p className="text-small text-text-secondary">Choose an available date, a location inside the NEER domain, and a supported model depth to request an explanation.</p></section> : null}
    {loading ? <section aria-live="polite" className={card + " py-12 text-center"}><div className="mx-auto mb-3 h-7 w-7 animate-spin rounded-full border-2 border-accent-300/30 border-t-accent-300" /><p className="text-small text-text-primary">Computing Integrated Gradients…</p><p className="mt-1 text-caption text-text-muted">Preparing the selected model input and integrating gradients along the baseline path.</p></section> : null}
    {currentError ? <section role="alert" className={card}><h2 className="font-medium text-text-primary">Unable to compute model explanation</h2><p className="mt-1 text-small text-text-secondary">{currentError}</p><button type="button" onClick={() => setRetryToken((value) => value + 1)} className="mt-3 inline-flex items-center gap-2 rounded-lg border border-border-subtle px-3 py-2 text-small text-text-secondary hover:bg-surface-800"><RefreshCw size={14} />Retry</button></section> : null}

    {current ? <>
      {current.partial ? <p role="status" className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-small text-amber-200">Partial attribution response: {current.missingFeatures.join(", ")} are unavailable and remain N/A.</p> : null}
      {current.data_mode === "DEMO_SYNTHETIC" ? <p role="status" className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-small text-amber-200"><AlertTriangle size={14} className="mr-1 inline" />The backend reports DEMO_SYNTHETIC data. These are actual model calculations, but not observational ocean evidence.</p> : null}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
        <section className={card}>
          <div className="mb-3 flex flex-wrap items-start justify-between gap-3"><div><h2 className="text-sm font-semibold uppercase tracking-wider text-text-primary">Integrated Gradients — Feature Attribution</h2><p className="mt-1 text-caption text-text-muted">Signed contribution · input channels summed across the model&apos;s spatial grid</p></div><span className="rounded-full border border-accent-400/30 px-2.5 py-1 text-[11px] text-accent-200">{current.integration_steps} integration steps</span></div>
          <AttributionChart features={current.features} selectedName={selectedFeature} onSelect={setSelectedFeature} />
          <p className="mt-3 text-center text-caption text-text-muted">Integrated Gradients Attribution (°C of decoder output)</p>
          <p className="mt-3 text-[11px] leading-5 text-text-muted">Positive and negative values describe direction relative to this baseline; neither label means good or bad. Attribution describes the model&apos;s behavior for this input, not physical causality.</p>
        </section>
        <aside className={card}>
          <h2 className="text-sm font-semibold uppercase tracking-wider text-text-primary">Prediction context</h2>
          <dl className="mt-3 grid grid-cols-2 gap-x-3 gap-y-3 border-b border-border-subtle pb-3 text-caption">
            <Detail label="Date" value={current.date} /><Detail label="Depth" value={formatDepth(current.depth)} />
            <Detail label="Requested location" value={formatLat(current.lat) + ", " + formatLon(current.lon)} />
            <Detail label="Climatology grid lookup" value={current.climatology_lat !== null && current.climatology_lon !== null ? formatLat(current.climatology_lat) + ", " + formatLon(current.climatology_lon) : "N/A"} />
            <Detail label="Model output · temperature" value={current.temperature === null ? "N/A" : current.temperature.toFixed(4) + " °C"} />
            <Detail label="Climatology" value={current.climatology === null ? "N/A" : current.climatology.toFixed(4) + " °C"} />
            <Detail label="Canonical anomaly" value={current.anomaly === null ? "N/A" : signed(current.anomaly) + " °C"} />
            <Detail label="Decoder output change from baseline" value={Number.isFinite(current.output_delta_from_baseline) ? current.output_delta_from_baseline.toFixed(6) + " °C" : "N/A"} />
            <Detail label="Data mode" value={current.data_mode || "N/A"} />
            <Detail label="Model version" value={current.model_version ?? "N/A"} />
            <Detail label="Checkpoint epoch" value={current.checkpoint_epoch ?? "N/A"} />
            <Detail label="IG completeness residual" value={current.completeness_error.toExponential(3) + " °C"} />
          </dl>
          <div className="mt-3"><p className="text-[10px] font-semibold uppercase tracking-wider text-text-muted">BASELINE</p><p className="mt-1 text-caption leading-5 text-text-secondary">{current.baseline}</p></div>
          {selected ? <div className="mt-4 rounded-lg border border-border-subtle bg-surface-950/50 p-3"><p className="text-[10px] font-semibold uppercase tracking-wider text-text-muted">SELECTED FEATURE</p><h3 className="mt-1 font-semibold text-text-primary">{selected.label}</h3><p className="text-caption text-text-muted">{selected.description}</p><p className="mt-2 font-mono text-small text-text-primary">{formatAttribution(selected.attribution)}{selected.attribution === null ? "" : " °C"}</p><p className="mt-1 text-caption text-text-secondary">{selected.attribution === null ? "Attribution unavailable." : selected.attribution > 0 ? "Positive contribution relative to baseline." : selected.attribution < 0 ? "Negative contribution relative to baseline." : "Zero contribution for this computed attribution."}</p></div> : null}
        </aside>
      </div>
      <section className={card}>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-text-primary">Feature attribution values</h2>
        <div className="overflow-x-auto"><table className="w-full min-w-[420px] text-left text-small"><thead className="text-caption uppercase tracking-wider text-text-muted"><tr><th className="pb-2">Feature</th><th className="pb-2">Input definition</th><th className="pb-2 text-right">Integrated Gradients attribution</th></tr></thead><tbody>{current.features.map((feature) => <tr key={feature.name} tabIndex="0" onFocus={() => setSelectedFeature(feature.name)} onClick={() => setSelectedFeature(feature.name)} className={"border-t border-border-subtle outline-none focus-visible:bg-surface-800 " + (selectedFeature === feature.name ? "bg-surface-800/60" : "")}><th scope="row" className="py-2 font-medium text-text-primary">{feature.label}</th><td className="py-2 text-caption text-text-muted">{feature.description}</td><td className="py-2 text-right font-mono text-text-primary">{formatAttribution(feature.attribution)}{feature.attribution === null ? "" : " °C"}</td></tr>)}</tbody></table></div>
      </section>
    </> : null}
    <nav aria-label="Related scientific views" className="flex flex-wrap gap-4 text-small">
      <Link className="text-accent-300 underline" href="/dashboard">View location on map</Link>
      <Link className="text-accent-300 underline" href="/vertical-profile">View vertical profile</Link>
      <Link className="text-accent-300 underline" href="/hovmoller">Open Hovmöller</Link>
    </nav>
  </main>;
}

function Meta({ label, value, emphasis = false }) {
  return <div className={card}><p className="text-[10px] font-semibold uppercase tracking-wider text-text-muted">{label}</p><p className={"mt-1 text-small " + (emphasis ? "font-semibold text-accent-200" : "text-text-primary")}>{value}</p></div>;
}
function Detail({ label, value }) { return <div><dt className="text-text-muted">{label}</dt><dd className="mt-0.5 break-words font-mono text-text-primary">{value}</dd></div>; }
function signed(value) { return (value > 0 ? "+" : "") + value.toFixed(4); }

function AttributionChart({ features, selectedName, onSelect }) {
  const values = features.map((feature) => feature.attribution).filter((value) => typeof value === "number" && Number.isFinite(value));
  const max = values.length ? Math.max(...values.map(Math.abs)) : 1;
  const left = 280; const right = 690; const center = (left + right) / 2; const half = (right - left) / 2;
  return <svg viewBox="0 0 710 330" className="h-auto w-full" role="group" aria-label="Signed Integrated Gradients attribution by feature. Negative values extend left of zero and positive values extend right.">
    {[left, center, right].map((x, index) => <g key={x}><line x1={x} x2={x} y1="18" y2="292" stroke={index === 1 ? "#94a3b8" : "rgba(148,163,184,.2)"} strokeWidth={index === 1 ? 1.5 : 1} strokeDasharray={index === 1 ? "0" : "3 5"} /><text x={x} y="316" textAnchor="middle" fill="#94a3b8" fontSize="11">{index === 0 ? (-max).toPrecision(3) : index === 1 ? "0" : max.toPrecision(3)}</text></g>)}
    {features.map((feature, index) => {
      const y = 34 + index * 36;
      const width = feature.attribution === null ? 0 : Math.abs(feature.attribution) / max * (half - 14);
      const x = feature.attribution < 0 ? center - width : center;
      const color = feature.attribution === null || feature.attribution === 0 ? "#94a3b8" : feature.attribution > 0 ? "#2dd4bf" : "#fb7185";
      return <g key={feature.name} tabIndex="0" role="button" aria-label={feature.label + ", attribution " + formatAttribution(feature.attribution)} aria-pressed={selectedName === feature.name} onFocus={() => onSelect(feature.name)} onMouseEnter={() => onSelect(feature.name)} onClick={() => onSelect(feature.name)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSelect(feature.name); } }} className="cursor-pointer outline-none">
        <text x="10" y={y + 13} fill="#cbd5e1" fontSize="12">{feature.label}</text>
        {feature.attribution === null ? <text x={center + 8} y={y + 13} fill="#94a3b8" fontSize="11">N/A</text> : <rect x={x} y={y} width={width || 1} height="18" rx="3" fill={color} stroke={selectedName === feature.name ? "#f8fafc" : "none"} strokeWidth="1.5"><title>{feature.label}: {formatAttribution(feature.attribution)} °C · {feature.attribution > 0 ? "Positive contribution" : feature.attribution < 0 ? "Negative contribution" : "Zero attribution"}</title></rect>}
        {feature.attribution !== null ? <text x={feature.attribution < 0 ? x - 7 : x + width + 7} y={y + 13} textAnchor={feature.attribution < 0 ? "end" : "start"} fill="#e2e8f0" fontSize="10">{formatAttribution(feature.attribution)}</text> : null}
      </g>;
    })}
  </svg>;
}
