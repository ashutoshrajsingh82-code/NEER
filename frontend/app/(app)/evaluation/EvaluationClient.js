"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, RefreshCw, ShieldCheck, TriangleAlert } from "lucide-react";
import { ApiError, argoEvaluation, evaluationMetrics } from "@/lib/api";
import { DEPTH_LEVELS } from "@/lib/oceanDomain";
import { EVALUATION_METRICS, EVALUATION_MODELS, validateArgoEvaluationResponse, validateEvaluationResponse } from "@/lib/evaluationResponse";

const panel = "rounded-2xl border border-border-subtle bg-surface-900/70 p-4 shadow-panel sm:p-5";
const MODEL_KEYS = EVALUATION_MODELS.map((model) => model.key);
const fmt = (value, digits = 3) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "N/A";

function validate(raw, fn, ...args) {
  const result = fn(raw, ...args);
  if (!result.ok) throw new ApiError({ code: "invalid_response", message: result.reason });
  return result.value;
}

function MetricChart({ metric, data, enabled, sourceLabel }) {
  const width = 720; const height = 250; const left = 56; const right = 18; const top = 20; const bottom = 42;
  const depths = data.depths ?? [];
  const observations = EVALUATION_MODELS.flatMap(({ key }) => enabled[key] ? depths.map((depth) => data.models?.[key]?.per_depth?.[depth]?.[metric.key]).filter((value) => Number.isFinite(value)) : []);
  let low = metric.key === "pearson" ? -1 : observations.length ? Math.min(...observations) : 0;
  let high = metric.key === "pearson" ? 1 : observations.length ? Math.max(...observations) : 1;
  if (metric.key === "bias" && observations.length) { low = Math.min(low, 0); high = Math.max(high, 0); }
  if (low === high) { low -= 0.5; high += 0.5; }
  const x = (depth) => left + (depths.length <= 1 ? (width - left - right) / 2 : (depth / Math.max(...depths)) * (width - left - right));
  const y = (value) => top + (high - value) / (high - low) * (height - top - bottom);
  const ticks = depths.length <= 5 ? depths : [depths[0], depths[Math.round((depths.length - 1) / 3)], depths[Math.round(2 * (depths.length - 1) / 3)], depths.at(-1)].filter((value, index, arr) => arr.indexOf(value) === index);
  const [hover, setHover] = useState(null);
  return <section className={panel}>
    <div className="mb-3"><h3 className="text-sm font-semibold text-text-primary">{metric.title} vs Depth</h3><p className="text-caption text-text-muted">{sourceLabel} · Depth (m) → · {metric.title}{metric.unit && data.units ? ` (${data.units})` : ""}</p></div>
    {!observations.length ? <div className="grid h-56 place-items-center text-small text-text-muted">Data unavailable for this metric.</div> : <div className="relative">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${metric.title} versus depth chart`} className="h-auto w-full overflow-visible">
        {[0, 0.25, 0.5, 0.75, 1].map((fraction) => { const value = high - fraction * (high - low); const yy = top + fraction * (height - top - bottom); return <g key={fraction}><line x1={left} x2={width - right} y1={yy} y2={yy} stroke={metric.key === "bias" && Math.abs(value) < (high - low) / 100 ? "#94a3b8" : "#243449"} strokeDasharray={metric.key === "bias" && Math.abs(value) < (high - low) / 100 ? "4 3" : undefined} /><text x={left - 8} y={yy + 4} textAnchor="end" fill="#94a3b8" fontSize="10">{fmt(value, 2)}</text></g>; })}
        {metric.key === "bias" && low <= 0 && high >= 0 ? <line x1={left} x2={width - right} y1={y(0)} y2={y(0)} stroke="#e2e8f0" strokeWidth="1.5" strokeDasharray="5 4" /> : null}
        {ticks.map((depth) => <g key={depth}><line x1={x(depth)} x2={x(depth)} y1={top} y2={height - bottom} stroke="#1e293b" /><text x={x(depth)} y={height - 18} textAnchor="middle" fill="#94a3b8" fontSize="10">{depth}</text></g>)}
        {EVALUATION_MODELS.filter(({ key }) => enabled[key]).map((model) => {
          let segment = [];
          const paths = [];
          depths.forEach((depth) => { const value = data.models?.[model.key]?.per_depth?.[depth]?.[metric.key]; if (Number.isFinite(value)) segment.push([depth, value]); else if (segment.length) { paths.push(segment); segment = []; } });
          if (segment.length) paths.push(segment);
          return <g key={model.key}>{paths.map((points, index) => <polyline key={index} fill="none" stroke={model.color} strokeWidth="2.4" strokeLinejoin="round" strokeLinecap="round" points={points.map(([depth, value]) => `${x(depth)},${y(value)}`).join(" ")} />)}
            {depths.map((depth) => { const value = data.models?.[model.key]?.per_depth?.[depth]?.[metric.key]; return Number.isFinite(value) ? <circle key={depth} cx={x(depth)} cy={y(value)} r="4" fill={model.color} stroke="#07111f" strokeWidth="1.5" tabIndex="0" aria-label={`${model.label}, depth ${depth} m, ${metric.title} ${value}${metric.unit && data.units ? ` ${data.units}` : ""}, source ${sourceLabel}`} onMouseEnter={() => setHover({ model: model.label, depth, value })} onMouseLeave={() => setHover(null)} onFocus={() => setHover({ model: model.label, depth, value })} onBlur={() => setHover(null)}><title>{`${model.label} · ${depth} m · ${metric.title}: ${value}${metric.unit && data.units ? ` ${data.units}` : ""} · ${sourceLabel}`}</title></circle> : null; })}</g>;
        })}
        <text x={width / 2} y={height - 2} textAnchor="middle" fill="#cbd5e1" fontSize="11">Depth (m)</text>
      </svg>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">{EVALUATION_MODELS.map((model) => enabled[model.key] ? <span key={model.key} className="inline-flex items-center gap-1.5 text-[11px] text-text-secondary"><i className="h-2 w-2 rounded-full" style={{ background: model.color }} />{model.label}</span> : null)}</div>
      {hover ? <p role="status" className="mt-2 rounded-md border border-border-subtle bg-surface-950 px-2.5 py-1.5 font-mono text-[11px] text-text-secondary">{hover.model} · {hover.depth} m · {metric.title}: {String(hover.value)}{metric.unit && data.units ? ` ${data.units}` : ""} · {sourceLabel}</p> : null}
    </div>}
  </section>;
}

function MetricCards({ data, depth, enabled }) {
  return <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">{EVALUATION_METRICS.map((metric) => <section key={metric.key} className={`${panel} p-3.5`}>
    <h3 className="text-xs font-semibold uppercase tracking-wider text-text-muted">{metric.title}{metric.unit && data.units ? ` (${data.units})` : ""}</h3>
    <div className="mt-2 space-y-1.5">{EVALUATION_MODELS.map((model) => <div key={model.key} className={`flex justify-between gap-2 text-small ${enabled[model.key] ? "" : "opacity-40"}`}><span className="text-text-secondary">{model.label}</span><span className="font-mono text-text-primary">{fmt(data.models?.[model.key]?.per_depth?.[depth]?.[metric.key])}{metric.key === "bias" && Number.isFinite(data.models?.[model.key]?.per_depth?.[depth]?.[metric.key]) ? "" : ""}</span></div>)}</div>
  </section>)}</div>;
}

function ArgoPanel({ data, loading, error, retry }) {
  return <section className={`${panel} border-sky-400/25`}>
    <div className="flex flex-wrap items-start justify-between gap-3"><div><h2 className="flex items-center gap-2 text-base font-semibold text-text-primary"><ShieldCheck size={17} className="text-sky-300" />{data?.isDemo ? "ARGO PIPELINE CHECK" : "ARGO INDEPENDENT OBSERVATIONAL VALIDATION"}</h2><p className="mt-1 text-small text-text-secondary">{data?.isDemo ? "Real ARGO observations are not configured. This labelled synthetic run demonstrates the pipeline only; it is not observational validation." : "Validation against independent ARGO observations. This protocol is separate from test/reanalysis scoring."}</p></div><span className="rounded-full border border-sky-300/30 bg-sky-300/5 px-2.5 py-1 text-[10px] font-semibold tracking-wider text-sky-200">{data?.isDemo ? "SYNTHETIC PIPELINE CHECK · NOT OBSERVATIONS" : "ARGO OBSERVATIONS"}</span></div>
    {loading ? <p role="status" className="mt-5 text-small text-text-muted">Loading independent ARGO validation…</p> : error ? <div role="alert" className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-amber-400/20 p-3"><p className="text-small text-text-secondary">ARGO validation: Unavailable. {error}</p><button onClick={retry} className="inline-flex items-center gap-2 rounded-md border border-border-subtle px-3 py-2 text-small"><RefreshCw size={14} />Retry</button></div> : !data ? <p className="mt-4 text-small text-text-muted">Data unavailable.</p> : <>
      {Array.isArray(data.warnings) && data.warnings.length ? <ul role="note" className="mt-4 space-y-1 rounded-lg border border-amber-400/25 bg-amber-400/5 p-3 text-caption text-amber-100">{data.warnings.map((warning, index) => <li key={index}>• {warning}</li>)}</ul> : null}
      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{EVALUATION_METRICS.map((metric) => <div key={metric.key} className="rounded-xl border border-border-subtle bg-surface-950/70 p-3"><h3 className="text-xs text-text-muted">{metric.title}</h3><p className="mt-2 font-mono text-lg text-text-primary">{fmt(data.metrics?.overall?.[metric.key === "pearson" ? "correlation" : metric.key])}</p><p className="text-[10px] text-text-muted">Pooled across matched depths; per-depth results are kept separate.</p></div>)}
      <div className="sm:col-span-2 lg:col-span-4"><dl className="mb-3 grid gap-2 rounded-lg border border-border-subtle bg-surface-950/60 p-3 text-caption sm:grid-cols-2 lg:grid-cols-3"><div><dt className="text-text-muted">{data.isDemo ? "Generated profile source" : "Observation source"}</dt><dd className="text-text-secondary">{data.argo?.source || "N/A"}</dd></div><div><dt className="text-text-muted">Matched profiles</dt><dd className="text-text-secondary">{data.counts?.matched_profiles ?? "N/A"}</dd></div><div><dt className="text-text-muted">Matched observations</dt><dd className="text-text-secondary">{data.metrics?.overall?.n_pairs ?? "N/A"}</dd></div><div><dt className="text-text-muted">Date range</dt><dd className="text-text-secondary">N/A (not provided by the validation report)</dd></div><div><dt className="text-text-muted">Geographic region</dt><dd className="text-text-secondary">N/A (not provided by the validation report)</dd></div><div><dt className="text-text-muted">Depth range</dt><dd className="text-text-secondary">{data.depth_coverage?.deepest_valid_depth_m ? `${data.depth_coverage.deepest_valid_depth_m.min}–${data.depth_coverage.deepest_valid_depth_m.max} m` : "N/A"}</dd></div><div className="sm:col-span-2 lg:col-span-3"><dt className="text-text-muted">{data.isDemo ? "Pipeline check details" : "Validation protocol"}</dt><dd className="break-words text-text-secondary">{data.banner || data.separation_note || JSON.stringify(data.config || {})}</dd></div></dl><div className="grid gap-3 lg:grid-cols-2">{EVALUATION_METRICS.map((metric) => <MetricChart key={metric.key} metric={metric} data={{ depths: data.depths, models: { neer: { per_depth: data.per_depth } }, units: "°C" }} enabled={{ neer: true, climatology: false, ridge: false, lightgbm: false }} sourceLabel={data.isDemo ? "Synthetic pipeline check · not observations" : "ARGO independent observational validation"} />)}</div></div>
    </div>
    </>}
  </section>;
}

export default function EvaluationClient() {
  const [split, setSplit] = useState("test");
  const [testState, setTestState] = useState({ loading: true, error: null, data: null });
  const [argoState, setArgoState] = useState({ loading: true, error: null, data: null });
  const [selectedDepth, setSelectedDepth] = useState(0);
  const [enabled, setEnabled] = useState(Object.fromEntries(MODEL_KEYS.map((key) => [key, true])));
  const requestRef = useRef(0);
  const testControllerRef = useRef(null);
  const argoRequestRef = useRef(0);

  const loadEvaluation = useCallback(async (selectedSplit = split) => {
    const id = ++requestRef.current;
    const controller = new AbortController();
    testControllerRef.current?.abort();
    testControllerRef.current = controller;
    setTestState({ loading: true, error: null, data: null });
    try {
      const data = validate(await evaluationMetrics({ split: selectedSplit }, { signal: controller.signal, timeoutMs: 120000 }), validateEvaluationResponse, selectedSplit);
      if (id === requestRef.current) { setTestState({ loading: false, error: null, data }); setSelectedDepth((current) => data.depths.includes(current) ? current : data.depths[0] ?? 0); }
    } catch (error) { if (id === requestRef.current) setTestState({ loading: false, error: error.message || "Evaluation data unavailable.", data: null }); }
    return () => controller.abort();
  }, [split]);

  const loadArgo = useCallback(async () => {
    const id = ++argoRequestRef.current;
    setArgoState({ loading: true, error: null, data: null });
    try {
      const raw = await argoEvaluation({ demo: false }, { timeoutMs: 120000 });
      const data = validate(raw, validateArgoEvaluationResponse);
      if (id === argoRequestRef.current) setArgoState({ loading: false, error: null, data });
    } catch (error) {
      if (id !== argoRequestRef.current) return;
      try {
        const raw = await argoEvaluation({ demo: true }, { timeoutMs: 120000 });
        const demoData = validate(raw, validateArgoEvaluationResponse, { allowDemo: true });
        if (id === argoRequestRef.current) setArgoState({ loading: false, error: null, data: demoData });
      } catch (demoError) {
        if (id === argoRequestRef.current) setArgoState({ loading: false, error: demoError.message || error.message || "ARGO data and pipeline check are unavailable.", data: null });
      }
    }
  }, []);

  useEffect(() => { loadEvaluation(split); }, [loadEvaluation, split]);
  useEffect(() => { loadArgo(); return () => { argoRequestRef.current += 1; }; }, [loadArgo]);
  useEffect(() => () => { requestRef.current += 1; testControllerRef.current?.abort(); }, []);

  const data = testState.data;
  const chartMetrics = useMemo(() => EVALUATION_METRICS, []);
  const synthetic = data?.source === "synthetic_demo";
  return <main className="mx-auto flex w-full max-w-[1500px] flex-col gap-5 p-4 sm:p-6 xl:p-8">
    <header className="flex flex-wrap items-start justify-between gap-4"><div><p className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-accent-300"><Activity size={14} />Model assessment</p><h1 className="mt-1 text-2xl font-semibold text-text-primary sm:text-3xl">Evaluation Dashboard</h1><p className="mt-2 max-w-3xl text-small text-text-secondary">Measured depth-wise metrics from the configured evaluation sources. Test/reanalysis, independent ARGO, and synthetic data remain separate.</p></div><button onClick={() => loadEvaluation(split)} disabled={testState.loading} className="inline-flex items-center gap-2 rounded-lg border border-border-subtle bg-surface-900 px-3 py-2 text-small text-text-secondary disabled:opacity-50"><RefreshCw size={14} className={testState.loading ? "animate-spin" : ""} />Refresh test results</button></header>

    <section className={panel}>
      <div className="flex flex-wrap items-end justify-between gap-4"><div><h2 className="text-base font-semibold text-text-primary">{synthetic ? "Evaluation results" : data ? "TEST / REANALYSIS EVALUATION" : "Evaluation results"}</h2><p className="mt-1 max-w-3xl text-small text-text-secondary">{synthetic ? "Synthetic pipeline measurements only; not real-world validation." : "Held-out model results are shown only when the backend provides measured evaluation data."}</p></div><div className="flex flex-wrap items-end gap-4"><label className="grid gap-1 text-caption text-text-muted">Evaluation split<select value={split} onChange={(event) => setSplit(event.target.value)} className="rounded-lg border border-border-subtle bg-surface-950 px-3 py-2 text-small text-text-primary"><option value="test">Held-out test</option><option value="val">Validation</option></select></label><label className="grid gap-1 text-caption text-text-muted">Depth<select value={selectedDepth} onChange={(event) => setSelectedDepth(Number(event.target.value))} className="rounded-lg border border-border-subtle bg-surface-950 px-3 py-2 text-small text-text-primary">{DEPTH_LEVELS.map((depth) => <option key={depth} value={depth}>{depth} m</option>)}</select></label></div></div>
      {testState.loading ? <p role="status" className="mt-5 text-small text-text-muted">Loading evaluation results…</p> : testState.error ? <div role="alert" className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-amber-400/20 p-3"><p className="text-small text-text-secondary">Evaluation data unavailable. {testState.error}</p><button onClick={() => loadEvaluation(split)} className="inline-flex items-center gap-2 rounded-md border border-border-subtle px-3 py-2 text-small"><RefreshCw size={14} />Retry</button></div> : synthetic ? <div className="mt-4 rounded-xl border-2 border-amber-400/50 bg-amber-400/5 p-4"><div className="mb-2 inline-flex rounded-full bg-amber-400 px-2.5 py-1 text-[10px] font-black tracking-widest text-slate-950">SYNTHETIC / DEMO · NOT REAL-WORLD VALIDATION</div><h3 className="font-semibold text-amber-100">Evaluation metrics</h3><p className="mt-1 text-small text-amber-100/80">{data.disclaimer || data.data_mode}. These scores are isolated from test/reanalysis and ARGO evidence.</p><div className="my-4"><MetricCards data={data} depth={selectedDepth} enabled={enabled} /></div><div className="grid gap-4 lg:grid-cols-2">{chartMetrics.map((metric) => <MetricChart key={metric.key} metric={metric} data={data} enabled={enabled} sourceLabel="SYNTHETIC / DEMO · NOT REAL-WORLD VALIDATION" />)}</div></div> : data ? <>
        <div className="mt-4 flex flex-wrap items-center gap-2"><span className="rounded-full border border-cyan-300/30 bg-cyan-300/5 px-2.5 py-1 text-[10px] font-semibold tracking-wider text-cyan-200">TEST / REANALYSIS</span><span className="text-caption text-text-muted">{data.data_mode} · {data.n_samples} samples · {data.aggregation}</span></div>
        {data.disclaimer ? <p className="mt-2 text-caption text-amber-200">{data.disclaimer}</p> : null}
        {Object.keys(data.unavailable ?? {}).length ? <p className="mt-2 text-caption text-text-muted">Unavailable model results: {Object.entries(data.unavailable).map(([name, reason]) => `${name}: ${reason}`).join(" · ")}</p> : null}
        <div className="mt-4 flex flex-wrap gap-2" aria-label="Model visibility">{EVALUATION_MODELS.map((model) => <button key={model.key} aria-pressed={enabled[model.key]} onClick={() => setEnabled((previous) => ({ ...previous, [model.key]: !previous[model.key] }))} className={`rounded-full border px-3 py-1.5 text-caption ${enabled[model.key] ? "border-border-subtle text-text-primary" : "border-border-subtle/50 text-text-muted opacity-60"}`}><i className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: model.color }} />{model.label}</button>)}</div>
        <div className="mt-4"><MetricCards data={data} depth={selectedDepth} enabled={enabled} /></div>
        {data.depths.length ? <div className="mt-4 grid gap-4 lg:grid-cols-2">{chartMetrics.map((metric) => <MetricChart key={metric.key} metric={metric} data={data} enabled={enabled} sourceLabel="Held-out test/reanalysis evaluation" />)}</div> : <p className="mt-4 text-small text-text-muted">No evaluation results available for this split.</p>}
        <div className="mt-4 overflow-x-auto rounded-xl border border-border-subtle"><table className="w-full min-w-[620px] text-left text-small"><caption className="p-3 text-left text-caption text-text-muted">Metrics at selected depth: {selectedDepth} m. Overall values are shown only where the backend provides an explicit pooled definition.</caption><thead className="border-b border-border-subtle bg-surface-950 text-caption text-text-muted"><tr><th className="p-3">Model</th>{EVALUATION_METRICS.map((metric) => <th key={metric.key} className="p-3">{metric.title}{metric.unit && data.units ? ` (${data.units})` : ""}</th>)}</tr></thead><tbody>{EVALUATION_MODELS.map((model) => <tr key={model.key} className="border-b border-border-subtle/60"><th className="p-3 font-medium text-text-primary">{model.label}</th>{EVALUATION_METRICS.map((metric) => <td key={metric.key} className="p-3 font-mono text-text-secondary">{fmt(data.models?.[model.key]?.per_depth?.[selectedDepth]?.[metric.key])}</td>)}</tr>)}</tbody></table></div>
        <p className="mt-3 text-caption text-text-muted">Dataset: {data.source_tensors_path || "N/A"} · Checkpoint: {data.checkpoint || "N/A"} · Period / last updated: N/A (not supplied by this API).</p>
        {data.invalidValues.length || data.invalidDepths ? <p role="status" className="mt-2 text-caption text-amber-200"><TriangleAlert size={13} className="mr-1 inline" />Invalid or unrecognized response values were excluded from plots.</p> : null}
      </> : <p className="mt-5 rounded-lg border border-border-subtle bg-surface-950/60 p-4 text-small text-text-muted">No held-out evaluation results are available for this split.</p>}
    </section>

    <ArgoPanel data={argoState.data} loading={argoState.loading} error={argoState.error} retry={loadArgo} />
    <section className={`${panel} border-dashed`}><div className="flex flex-wrap items-start justify-between gap-3"><div><p className="text-small text-text-secondary">Synthetic values are not evidence of real-world model performance and are never combined with held-out evaluation or ARGO observations.</p></div><span className="rounded-full border border-amber-300/30 px-2.5 py-1 text-[10px] font-semibold tracking-wider text-amber-200">NOT REAL-WORLD VALIDATION</span></div><p className="mt-3 text-small text-text-muted">The generated pipeline check is displayed separately and does not use observational profiles.</p></section>
  </main>;
}
