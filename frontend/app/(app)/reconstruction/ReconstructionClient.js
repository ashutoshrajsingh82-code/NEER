"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Activity, ArrowUpRight, Download, RefreshCw, Waves } from "lucide-react";
import { ErrorState, LoadingSkeleton, MetricCard, Panel } from "@/components/ui";
import { profile as fetchProfile } from "@/lib/api";
import { formatDate, formatSigned } from "@/lib/format";
import { formatDepth, formatLat, formatLon } from "@/lib/oceanDomain";
import { profilePointAtDepth, validateProfileResponse } from "@/lib/profileResponse";
import { useApiRequest } from "@/lib/useApiRequest";
import { useNeerContext } from "../dashboard/_context/NeerContext";

const panelClass = "rounded-xl border border-border-subtle bg-surface-900/80 p-4 sm:p-5";
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const shown = (value, digits = 2) => finite(value) ? Number(value).toFixed(digits) : "N/A";

async function loadValidatedProfile(params) {
  const response = await fetchProfile(params);
  const checked = validateProfileResponse(response, params);
  if (!checked.ok) throw new Error(checked.reason);
  return checked.value;
}

function ProfileChart({ profile }) {
  const allValues = [...(profile.temperature ?? []), ...(profile.climatology ?? [])].filter(finite);
  if (!allValues.length) return <p className="py-8 text-small text-text-muted">No finite profile values were returned for this selection.</p>;

  const min = Math.floor(Math.min(...allValues) - 1);
  const max = Math.ceil(Math.max(...allValues) + 1);
  const range = Math.max(1, max - min);
  const left = 74; const right = 790; const top = 24; const bottom = 322;
  const y = (depth) => top + ((depth - profile.depths[0]) / Math.max(1, profile.depths.at(-1) - profile.depths[0])) * (bottom - top);
  const x = (value) => left + ((value - min) / range) * (right - left);
  const makePath = (values) => {
    let drawing = false;
    return profile.depths.map((depth, index) => {
      const value = values?.[index];
      if (!finite(value)) { drawing = false; return ""; }
      const command = `${drawing ? "L" : "M"}${x(value)},${y(depth)}`;
      drawing = true;
      return command;
    }).filter(Boolean).join(" ");
  };
  const xTicks = Array.from({ length: 5 }, (_, index) => min + range * index / 4);
  const depthsToLabel = [...new Set([profile.depths[0], 100, 500, profile.depths.at(-1)].filter((depth) => profile.depths.includes(depth)))];

  return <div className="w-full overflow-x-auto">
    <svg viewBox="0 0 820 370" role="img" aria-label="Reconstructed temperature and climatology by depth" className="h-auto min-w-[560px] w-full">
      {xTicks.map((tick, index) => <g key={`x-${index}`}><line x1={x(tick)} x2={x(tick)} y1={top} y2={bottom} stroke="#334155" strokeDasharray="3 5" /><text x={x(tick)} y={bottom + 24} textAnchor="middle" fill="#94a3b8" fontSize="12">{tick.toFixed(1)}°C</text></g>)}
      {depthsToLabel.map((depth) => <g key={`d-${depth}`}><line x1={left} x2={right} y1={y(depth)} y2={y(depth)} stroke="#334155" strokeDasharray="3 5" /><text x={left - 10} y={y(depth) + 4} textAnchor="end" fill="#94a3b8" fontSize="12">{depth} m</text></g>)}
      <line x1={left} x2={left} y1={top} y2={bottom} stroke="#94a3b8" /><line x1={left} x2={right} y1={bottom} y2={bottom} stroke="#94a3b8" />
      <path d={makePath(profile.temperature)} fill="none" stroke="#2dd4bf" strokeWidth="3" />
      <path d={makePath(profile.climatology)} fill="none" stroke="#fbbf24" strokeWidth="2.5" strokeDasharray="7 5" />
      {profile.depths.map((depth, index) => finite(profile.temperature?.[index]) ? <circle key={`t-${depth}`} cx={x(profile.temperature[index])} cy={y(depth)} r="3.5" fill="#2dd4bf"><title>{`${depth} m · model ${profile.temperature[index].toFixed(2)} °C`}</title></circle> : null)}
      <text x={(left + right) / 2} y="366" textAnchor="middle" fill="#cbd5e1" fontSize="12">Temperature (°C)</text>
      <text x="18" y={(top + bottom) / 2} textAnchor="middle" transform={`rotate(-90 18 ${(top + bottom) / 2})`} fill="#cbd5e1" fontSize="12">Depth (m), increasing downward</text>
    </svg>
    <div className="flex flex-wrap gap-4 px-2 pb-2 text-caption text-text-secondary"><span><i className="mr-2 inline-block w-5 border-t-2 border-teal-400" />MODEL OUTPUT</span><span><i className="mr-2 inline-block w-5 border-t-2 border-dashed border-amber-300" />CLIMATOLOGY</span></div>
  </div>;
}

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export default function ReconstructionClient() {
  const { selectedPoint, date, depth, pointStatus, pointInspection, modelInfoData, dataMode } = useNeerContext();
  const requestProfile = useCallback(loadValidatedProfile, []);
  const { data: profileData, error: profileError, isLoading: profileLoading, isError: profileIsError, run: runProfile, reset: resetProfile } = useApiRequest(requestProfile);
  const [profileKey, setProfileKey] = useState(null);
  const selectionKey = selectedPoint && date ? `${date}|${selectedPoint.lat}|${selectedPoint.lon}` : null;
  const reconstruction = pointInspection?.data;
  const reconstructionMatches = Boolean(reconstruction && selectedPoint && reconstruction.date === date && reconstruction.depth === depth && reconstruction.lat === selectedPoint.lat && reconstruction.lon === selectedPoint.lon);

  useEffect(() => {
    if (!selectionKey || (pointStatus && pointStatus !== "ocean")) {
      resetProfile();
      setProfileKey(null);
      return undefined;
    }
    const params = { lat: selectedPoint.lat, lon: selectedPoint.lon, date };
    setProfileKey(selectionKey);
    runProfile(params).catch(() => {});
  }, [selectionKey, pointStatus, selectedPoint?.lat, selectedPoint?.lon, date, resetProfile, runProfile]);

  const profile = profileData && profileKey === selectionKey && profileData.date === date && profileData.lat === selectedPoint?.lat && profileData.lon === selectedPoint?.lon ? profileData : null;
  const selectedLevel = useMemo(() => profile ? profilePointAtDepth(profile, depth) : null, [profile, depth]);
  const checkpoint = modelInfoData?.checkpoint ?? {};
  const effectiveDataMode = reconstruction?.data_mode ?? dataMode ?? "N/A";

  const exportJson = () => {
    if (!reconstructionMatches) return;
    saveBlob(new Blob([JSON.stringify({
      metadata: { title: "NEER point reconstruction", source: "NEER backend", data_mode: effectiveDataMode },
      selection: { date, latitude: selectedPoint.lat, longitude: selectedPoint.lon, depth },
      point_result: reconstruction,
      vertical_profile: profile,
    }, null, 2)], { type: "application/json" }), `neer-reconstruction-${date}-${depth}m.json`);
  };

  const exportCsv = () => {
    if (!profile) return;
    const lines = [
      `# source,NEER backend reconstruction`, `# date,${profile.date}`, `# latitude,${profile.lat}`, `# longitude,${profile.lon}`, `# data_mode,${effectiveDataMode}`,
      "depth_m,model_output_temperature_c,climatology_temperature_c,anomaly_c",
      ...profile.depths.map((level, index) => [level, profile.temperature[index] ?? "", profile.climatology?.[index] ?? "", profile.anomaly[index] ?? ""].join(",")),
    ];
    saveBlob(new Blob([lines.join("\r\n")], { type: "text/csv;charset=utf-8" }), `neer-profile-${date}-${profile.lat}-${profile.lon}.csv`);
  };

  if (pointStatus && pointStatus !== "ocean") return <main className="mx-auto w-full max-w-6xl space-y-5 p-4 sm:p-6"><ErrorState title="Reconstruction unavailable at this location" message="Select an ocean grid point inside the NEER domain to view a reconstruction." /><Link className="text-small text-accent-300 underline" href="/dashboard">Choose an ocean location</Link></main>;

  return <main className="mx-auto flex w-full max-w-7xl flex-col gap-5 p-4 sm:p-6 xl:p-8">
    <header className="flex flex-wrap items-start justify-between gap-4">
      <div><p className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-accent-300"><Activity size={14} />NEER · Reconstruction</p><h1 className="mt-1 text-2xl font-semibold text-text-primary sm:text-3xl">Reconstruction Results</h1><p className="mt-2 max-w-3xl text-small text-text-secondary">Backend model output for the current date, ocean location, and selected depth, with the complete vertical profile and run context.</p></div>
      <div className="flex flex-wrap gap-2"><button type="button" onClick={() => pointInspection?.retry?.()} disabled={!selectedPoint || !date || pointInspection?.isLoading} className="inline-flex items-center gap-2 rounded-lg border border-border-subtle px-3 py-2 text-small text-text-primary disabled:cursor-not-allowed disabled:opacity-50"><RefreshCw size={14} className={pointInspection?.isLoading ? "animate-spin" : ""} />{pointInspection?.isLoading ? "Reconstructing…" : "Run again"}</button><button type="button" onClick={exportJson} disabled={!reconstructionMatches} className="inline-flex items-center gap-2 rounded-lg border border-border-subtle px-3 py-2 text-small text-text-primary disabled:cursor-not-allowed disabled:opacity-50"><Download size={14} />Export JSON</button><button type="button" onClick={exportCsv} disabled={!profile} className="inline-flex items-center gap-2 rounded-lg bg-accent-600 px-3 py-2 text-small font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"><Download size={14} />Export profile CSV</button></div>
    </header>

    <section className={panelClass} aria-label="Reconstruction selection">
      <div className="mb-3 flex items-center gap-2"><Waves size={16} className="text-cyan-300" /><h2 className="text-sm font-semibold uppercase tracking-wider text-text-primary">Current reconstruction context</h2></div>
      <div className="grid gap-3 text-small sm:grid-cols-2 xl:grid-cols-4">
        <Context label="Date" value={date ? formatDate(date) : "Loading available dates…"} />
        <Context label="Location" value={selectedPoint ? `${formatLat(selectedPoint.lat)}, ${formatLon(selectedPoint.lon)}` : "Loading default ocean location…"} />
        <Context label="Selected depth" value={formatDepth(depth)} />
        <Context label="Data mode" value={effectiveDataMode} />
      </div>
      {!selectedPoint || !date || depth == null ? <p role="status" className="mt-3 text-caption text-text-muted">Waiting for the shared backend date, depth, and default ocean location.</p> : null}
      {pointInspection?.isLoading ? <p role="status" className="mt-3 text-caption text-cyan-200">The backend is reconstructing the selected ocean value…</p> : null}
      {pointInspection?.isError ? <div className="mt-3"><ErrorState title="Reconstruction failed" message={pointInspection.error?.message || "The backend could not reconstruct this selection."} onRetry={() => pointInspection.retry?.()} /></div> : null}
      {reconstructionMatches ? <p role="status" className="mt-3 text-caption text-emerald-200">Reconstruction complete · {shown(reconstruction.lat, 2)}°N, {shown(reconstruction.lon, 2)}°E · {formatDate(reconstruction.date)} · {formatDepth(reconstruction.depth)}</p> : null}
    </section>

    {effectiveDataMode === "DEMO_SYNTHETIC" ? <p role="note" className="rounded-lg border border-amber-400/30 bg-amber-400/5 p-3 text-caption text-amber-100">This result comes from DEMO_SYNTHETIC data. It demonstrates the reconstruction pipeline and is not observational evidence or independent scientific validation.</p> : null}

    {pointInspection?.isLoading && !reconstructionMatches ? <LoadingSkeleton variant="panel" label="Loading reconstruction result" /> : null}
    {reconstructionMatches ? <>
      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4" aria-label="Reconstruction values">
        <MetricCard title="MODEL OUTPUT" value={shown(reconstruction.temperature)} unit={finite(reconstruction.temperature) ? "°C" : undefined} description="Reconstructed temperature at selected depth" />
        <MetricCard title="CLIMATOLOGY" value={shown(reconstruction.climatology)} unit={finite(reconstruction.climatology) ? "°C" : undefined} description="Backend reference, when available" />
        <MetricCard title="ANOMALY" value={finite(reconstruction.anomaly) ? formatSigned(reconstruction.anomaly) : "N/A"} unit={finite(reconstruction.anomaly) ? "°C" : undefined} description="MODEL OUTPUT − CLIMATOLOGY" />
        <MetricCard title="BACKEND LATENCY" value={shown(reconstruction.latency_ms, 1)} unit="ms" description={reconstruction.cache_hit ? "Cached backend result" : "Model request"} />
      </section>

      <Panel emphasis="base" icon={Waves} title="Vertical profile" subtitle={profile ? `${profile.depths.length} backend depth levels · ${formatDate(profile.date)}` : "Loading all available model depths"} bodyClassName="flex flex-col gap-3">
        {profileLoading && !profile ? <LoadingSkeleton variant="panel" label="Loading complete vertical profile" /> : null}
        {profileIsError ? <ErrorState title="Vertical profile unavailable" message={profileError?.message || "The point reconstruction is available, but its full-depth profile could not be loaded."} onRetry={() => selectedPoint && date ? runProfile({ lat: selectedPoint.lat, lon: selectedPoint.lon, date }).catch(() => {}) : null} /> : null}
        {profile ? <>
          <ProfileChart profile={profile} />
          <div className="overflow-x-auto rounded-lg border border-border-subtle"><table className="w-full min-w-[560px] text-left text-small"><caption className="p-3 text-left text-caption text-text-muted">Exact values returned by the backend at each supported model depth; no interpolation is applied.</caption><thead className="bg-surface-950 text-caption text-text-muted"><tr><th className="p-3">Depth</th><th className="p-3">MODEL OUTPUT</th><th className="p-3">CLIMATOLOGY</th><th className="p-3">ANOMALY</th><th className="p-3">Selected</th></tr></thead><tbody>{profile.depths.map((level, index) => <tr key={level} className={`border-t border-border-subtle ${level === depth ? "bg-accent-900/30" : ""}`}><th scope="row" className="p-3 font-mono font-medium text-text-primary">{formatDepth(level)}</th><td className="p-3 font-mono">{finite(profile.temperature[index]) ? `${profile.temperature[index].toFixed(2)} °C` : "N/A"}</td><td className="p-3 font-mono">{finite(profile.climatology?.[index]) ? `${profile.climatology[index].toFixed(2)} °C` : "N/A"}</td><td className="p-3 font-mono">{finite(profile.anomaly[index]) ? `${formatSigned(profile.anomaly[index])} °C` : "N/A"}</td><td className="p-3">{level === depth ? <span className="rounded-full bg-accent-800 px-2 py-1 text-[10px] font-semibold text-accent-100">CURRENT</span> : "—"}</td></tr>)}</tbody></table></div>
          <div className="grid gap-3 sm:grid-cols-3"><MetricCard title={`MODEL OUTPUT @ ${formatDepth(depth)}`} value={selectedLevel?.neer === "N/A" ? "N/A" : shown(Number(selectedLevel?.neer))} unit={selectedLevel?.neer !== "N/A" ? "°C" : undefined} /><MetricCard title="PROFILE SURFACE VALUE" value={shown(profile.temperature[0])} unit={finite(profile.temperature[0]) ? "°C" : undefined} /><MetricCard title={`PROFILE ANOMALY @ ${formatDepth(depth)}`} value={selectedLevel?.anomaly === "N/A" ? "N/A" : selectedLevel?.anomaly == null ? "N/A" : formatSigned(Number(selectedLevel.anomaly))} unit={finite(Number(selectedLevel?.anomaly)) ? "°C" : undefined} /></div>
        </> : null}
      </Panel>

      <div className="grid gap-4 xl:grid-cols-2">
        <section className={panelClass}><h2 className="text-sm font-semibold uppercase tracking-wider text-text-primary">Model and run details</h2><dl className="mt-3 grid gap-3 sm:grid-cols-2"> <Context label="Model checkpoint" value={checkpoint.filename || checkpoint.path?.split(/[\\/]/).pop() || "N/A"} /><Context label="Checkpoint epoch" value={checkpoint.epoch == null ? "N/A" : String(checkpoint.epoch)} /><Context label="Embedding dimensions" value={reconstruction.embedding_dim == null ? "N/A" : String(reconstruction.embedding_dim)} /><Context label="Backend mode" value={reconstruction.mode || "N/A"} /><Context label="Result cache" value={reconstruction.cache_hit ? "Hit" : "Computed"} /><Context label="Backend latency" value={`${shown(reconstruction.latency_ms, 1)} ms`} /></dl></section>
        <section className={panelClass}><h2 className="text-sm font-semibold uppercase tracking-wider text-text-primary">Interpretation and provenance</h2><p className="mt-3 text-small leading-6 text-text-secondary">The model output is a reconstructed value for the selected grid point, date, and depth. The anomaly is supplied by the backend and follows its model-output-minus-climatology definition. Missing reference values remain N/A.</p><p className="mt-3 text-caption text-text-muted">Data mode: {effectiveDataMode} · Backend notes: {reconstruction.notes?.length ? reconstruction.notes.join(" ") : "No additional notes supplied."}</p>{!finite(reconstruction.climatology) ? <p className="mt-3 rounded-lg border border-amber-400/20 bg-amber-400/5 p-3 text-caption text-amber-100">Climatology was unavailable for this selection, so anomaly is also unavailable.</p> : null}</section>
      </div>
    </> : null}

    <nav aria-label="Related scientific views" className="flex flex-wrap gap-4 text-small"><Link className="inline-flex items-center gap-1 text-accent-300 underline" href="/dashboard">View on map <ArrowUpRight size={13} /></Link><Link className="text-accent-300 underline" href="/vertical-profile">Vertical profile</Link><Link className="text-accent-300 underline" href="/hovmoller">Hovmöller</Link><Link className="text-accent-300 underline" href="/3d-ocean">3D Ocean</Link><Link className="text-accent-300 underline" href="/explainability">Explainability</Link></nav>
  </main>;
}

function Context({ label, value }) {
  return <div className="min-w-0"><dt className="text-[10px] font-semibold uppercase tracking-wider text-text-muted">{label}</dt><dd className="mt-1 break-words font-mono text-small text-text-primary">{value ?? "N/A"}</dd></div>;
}
