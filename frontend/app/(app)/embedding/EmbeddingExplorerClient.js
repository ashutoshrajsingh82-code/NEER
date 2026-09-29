"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Download, Network, RefreshCw } from "lucide-react";
import { dates as fetchDates, getEmbedding, modelInfo as fetchModelInfo } from "@/lib/api";
import { useDateDepthContext } from "../dashboard/_context/DateDepthContext";
import { monthOfDate, normalizeEmbeddingDataset, regionColor } from "@/lib/embeddingExplorer";
import EmbeddingScatter from "@/components/embedding/EmbeddingScatter";

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const EMPTY_RECORDS = [];
const selectClass = "rounded-lg border border-border-subtle bg-surface-900 px-3 py-2 text-small text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-300";
const cardClass = "rounded-xl border border-border-subtle bg-surface-900/70 p-4";

async function loadEmbeddings(signal, onProgress, loadIdentity) {
  onProgress({ stage: "Loading available dates…", completed: 0, total: 0 });
  // `dates()` returns the validated API payload directly (unlike the
  // result-wrapped helpers used by the dashboard context).
  const datesPayload = await fetchDates({ signal });
  if (!Array.isArray(datesPayload?.dates)) {
    throw new Error("The backend returned an invalid available-dates response.");
  }
  const dateList = datesPayload.dates;
  if (!dateList.length) return { records: [], dimension: null, invalidCount: 0, invalidReasons: [], warning: null, expectedDimensionMismatch: false, dates: [], dataMode: datesPayload.dataMode, isSynthetic: datesPayload.isSynthetic };
  const results = new Array(dateList.length);
  let next = 0;
  let completed = 0;
  const workers = Array.from({ length: Math.min(4, dateList.length) }, async () => {
    while (next < dateList.length) {
      if (signal.aborted) throw new DOMException("Aborted", "AbortError");
      const index = next++;
      results[index] = await getEmbedding({ date: dateList[index] }, { signal });
      completed += 1;
      onProgress({ stage: "Loading actual model embeddings…", completed, total: dateList.length });
    }
  });
  await Promise.all(workers);
  const normalized = normalizeEmbeddingDataset(results, dateList);
  if (!normalized.ok) throw new Error(normalized.reason);
  if (!normalized.records.length) {
    throw new Error(normalized.invalidReasons[0]?.reason || "The backend returned no usable model embeddings.");
  }
  const modeResponse = results.find((item) => item && typeof item.data_mode === "string");
  const identity = String(loadIdentity) + ":" + dateList.length + ":" + dateList[0] + ":" + dateList[dateList.length - 1];
  return { ...normalized, dates: dateList, identity, dataMode: modeResponse?.data_mode ?? datesPayload.dataMode ?? null, isSynthetic: datesPayload.isSynthetic === true };
}

function getVersion(info) {
  return info?.version ?? info?.model_version ?? info?.checkpoint?.version ?? info?.checkpoint?.model_version ?? null;
}

export default function EmbeddingExplorerClient() {
  const { selectDate } = useDateDepthContext();
  const [dataset, setDataset] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loadProgress, setLoadProgress] = useState({ stage: "Loading available dates…", completed: 0, total: 0 });
  const [loadError, setLoadError] = useState("");
  const [retryToken, setRetryToken] = useState(0);
  const [method, setMethod] = useState("pca");
  const [colorMode, setColorMode] = useState("month");
  const [perplexity, setPerplexity] = useState(30);
  const [iterations, setIterations] = useState(1000);
  const [projection, setProjection] = useState(null);
  const [projectionToken, setProjectionToken] = useState(0);
  const [projectionState, setProjectionState] = useState({ status: "idle", stage: "", progress: 0, error: "" });
  const [selectedId, setSelectedId] = useState(null);
  const [vectorExpanded, setVectorExpanded] = useState(false);
  const cacheRef = useRef(new Map());
  const workerRef = useRef(null);
  const computationIdRef = useRef(0);
  const startLoad = useCallback(() => setRetryToken((value) => value + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    setDataset(null);
    setLoadError("");
    setLoadProgress({ stage: "Loading available dates…", completed: 0, total: 0 });
    Promise.all([
      loadEmbeddings(controller.signal, setLoadProgress, retryToken),
      fetchModelInfo({ signal: controller.signal }).catch(() => null),
    ]).then(([result, info]) => {
      if (controller.signal.aborted) return;
      setDataset(result);
      setModelInfo(info);
      setSelectedId((current) => result.records.some((record) => record.id === current) ? current : result.records[0]?.id ?? null);
    }).catch((error) => {
      if (controller.signal.aborted || error?.name === "AbortError") return;
      controller.abort();
      setLoadError(error?.message || "Unable to load embeddings.");
      setDataset(null);
    });
    return () => controller.abort();
  }, [retryToken]);

  const records = dataset?.records ?? EMPTY_RECORDS;
  const selected = records.find((record) => record.id === selectedId) ?? null;
  const perplexityMax = Math.max(2, Math.min(50, records.length - 1));
  const effectivePerplexity = Math.min(perplexity, Math.max(2, records.length - 1));

  useEffect(() => {
    if (!dataset || records.length < 2) {
      setProjection(null);
      setProjectionState({ status: "idle", stage: "", progress: 0, error: "" });
      return undefined;
    }
    const actualPerplexity = Math.max(2, Math.min(effectivePerplexity, records.length - 1));
    const cacheKey = dataset.identity + ":" + (method === "pca" ? "pca" : "tsne:" + actualPerplexity + ":" + iterations + ":42");
    const cached = cacheRef.current.get(cacheKey);
    if (cached) {
      setProjection({ ...cached, method });
      setProjectionState({ status: "ready", stage: "Projection ready", progress: 1, error: "" });
      return undefined;
    }
    if (workerRef.current) workerRef.current.terminate();
    const id = ++computationIdRef.current;
    const worker = new Worker(new URL("../../../lib/embeddingProjection.worker.js", import.meta.url));
    workerRef.current = worker;
    setProjection(null);
    setProjectionState({ status: "computing", stage: method === "pca" ? "Preparing embeddings…" : "Preparing t-SNE…", progress: 0, error: "" });
    worker.onmessage = (event) => {
      if (id !== computationIdRef.current) return;
      const message = event.data;
      if (message.type === "progress") setProjectionState({ status: "computing", stage: message.stage, progress: message.progress, error: "" });
      if (message.type === "result") {
        const value = { ...message.result, method, records };
        cacheRef.current.set(cacheKey, value);
        setProjection(value);
        setProjectionState({ status: "ready", stage: "Projection ready", progress: 1, error: "" });
        worker.terminate();
        if (workerRef.current === worker) workerRef.current = null;
      }
      if (message.type === "error") {
        setProjectionState({ status: "error", stage: "", progress: 0, error: message.message });
        worker.terminate();
        if (workerRef.current === worker) workerRef.current = null;
      }
    };
    worker.onerror = (event) => {
      if (id === computationIdRef.current) setProjectionState({ status: "error", stage: "", progress: 0, error: event.message || "Projection computation failed." });
      worker.terminate();
    };
    worker.postMessage({ id, method, matrix: records.map((record) => record.embedding), parameters: { perplexity: actualPerplexity, iterations, seed: 42 } });
    return () => {
      computationIdRef.current += 1;
      worker.terminate();
      if (workerRef.current === worker) workerRef.current = null;
    };
  }, [dataset, method, effectivePerplexity, iterations, records, projectionToken]);

  const points = useMemo(() => projection?.coordinates?.map(([x, y], index) => ({ x, y, record: records[index] })).filter((point) => point.record && Number.isFinite(point.x) && Number.isFinite(point.y)) ?? [], [projection, records]);
  const selectedPoint = points.find((point) => point.record.id === selectedId);
  const axisLabels = projection?.axes ?? (method === "pca" ? ["PC1", "PC2"] : ["t-SNE 1", "t-SNE 2"]);
  const availableMeta = {
    latitude: records.some((record) => Number.isFinite(record.latitude)),
    longitude: records.some((record) => Number.isFinite(record.longitude)),
    anomaly: records.some((record) => Number.isFinite(record.anomaly)),
    region: records.some((record) => Boolean(record.region)),
  };
  const selectedMonth = monthOfDate(selected?.date);
  const version = getVersion(modelInfo);

  const exportSelected = () => {
    if (!selected || !selectedPoint) return;
    const payload = {
      observation_id: selected.id, date: selected.date, latitude: selected.latitude, longitude: selected.longitude,
      projection_method: projection.method.toUpperCase(), projection_x: selectedPoint.x, projection_y: selectedPoint.y,
      color_variable: colorMode, embedding_dimension: selected.dimension, embedding: selected.embedding,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = "neer-embedding-" + selected.date + ".json"; anchor.click();
    URL.revokeObjectURL(url);
  };

  const colorOptions = [
    ["month", "Month", true], ["latitude", "Latitude", availableMeta.latitude], ["longitude", "Longitude", availableMeta.longitude],
    ["anomaly", "Temperature anomaly", availableMeta.anomaly], ["region", "Region", availableMeta.region],
  ];

  return <div className="mx-auto flex max-w-[1500px] flex-col gap-5">
    <header className="flex flex-wrap items-start justify-between gap-4">
      <div><div className="mb-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-accent-300"><Network size={15} /> NEER · Representation analysis</div>
        <h1 className="text-2xl font-semibold tracking-tight text-text-primary">Embedding Explorer</h1>
        <p className="mt-2 max-w-3xl text-small leading-6 text-text-secondary">Explore actual domain-pooled model vectors returned by the backend for available dates. Projection axes describe embedding space, not geographic coordinates.</p>
      </div>
      <button type="button" onClick={startLoad} className="inline-flex items-center gap-2 rounded-lg border border-border-subtle px-3 py-2 text-small text-text-secondary hover:bg-surface-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-300"><RefreshCw size={15} />Reload data</button>
    </header>

    <section className={cardClass} aria-label="Projection controls">
      <div className="flex flex-wrap items-end gap-4">
        <label className="flex flex-col gap-1.5 text-caption text-text-muted">PROJECTION
          <select aria-label="Projection method" value={method} onChange={(event) => setMethod(event.target.value)} className={selectClass}><option value="pca">PCA</option><option value="tsne" disabled={records.length < 3}>t-SNE</option></select>
        </label>
        <label className="flex flex-col gap-1.5 text-caption text-text-muted">COLOR BY
          <select aria-label="Color by" value={colorMode} onChange={(event) => setColorMode(event.target.value)} className={selectClass}>
            {colorOptions.map(([value, label, available]) => <option key={value} value={value}>{label}{!available ? " — Unavailable" : ""}</option>)}
          </select>
        </label>
        {method === "tsne" ? <>
          <label className="flex flex-col gap-1.5 text-caption text-text-muted">PERPLEXITY
            <input aria-label="t-SNE perplexity" className={selectClass} type="number" min="2" max={perplexityMax} value={perplexity} onChange={(event) => setPerplexity(Math.min(perplexityMax, Math.max(2, Number(event.target.value) || 2)))} />
            <span className="text-[10px]">Default 30; effective value is capped below the observation count ({effectivePerplexity}).</span>
          </label>
          <label className="flex flex-col gap-1.5 text-caption text-text-muted">ITERATIONS
            <select aria-label="t-SNE iterations" className={selectClass} value={iterations} onChange={(event) => setIterations(Number(event.target.value))}><option value="500">500</option><option value="1000">1000 (default)</option><option value="1500">1500</option></select>
            <span className="text-[10px]">Seed 42 · layout changes with parameters.</span>
          </label>
        </> : null}
      </div>
    </section>

    {dataset?.isSynthetic ? <div role="status" className="flex gap-2 rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-small text-amber-200"><AlertTriangle size={16} className="mt-0.5 shrink-0" /><span>The backend reports DEMO_SYNTHETIC data. These are actual model responses for that dataset, but are not observational ocean evidence.</span></div> : null}
    {loadError ? <section role="alert" className={cardClass}><h2 className="font-medium text-text-primary">Unable to load embeddings</h2><p className="mt-1 text-small text-text-secondary">{loadError}</p><button type="button" onClick={startLoad} className="mt-3 rounded-lg bg-accent-500 px-3 py-2 text-small font-medium text-surface-950">Retry loading</button></section> : null}
    {!dataset && !loadError ? <section aria-live="polite" className={cardClass + " py-12 text-center"}><div className="mx-auto mb-3 h-7 w-7 animate-spin rounded-full border-2 border-accent-300/30 border-t-accent-300" /><p className="text-small text-text-primary">{loadProgress.stage || "Loading embeddings…"}</p>{loadProgress.total ? <p className="mt-1 text-caption text-text-muted">{loadProgress.completed.toLocaleString()} / {loadProgress.total.toLocaleString()} dates</p> : null}</section> : null}
    {dataset && !records.length ? <section className={cardClass}><h2 className="font-medium text-text-primary">No embeddings available.</h2><p className="mt-1 text-small text-text-secondary">The backend did not return valid, dimension-consistent vectors for available dates.</p></section> : null}

    {dataset && records.length ? <>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Metric label="VALID OBSERVATIONS" value={records.length.toLocaleString()} /><Metric label="EMBEDDING DIMENSION" value={records[0].dimension + " dimensions"} />
        <Metric label="PROJECTION" value={method === "pca" ? "PCA" : "t-SNE"} /><Metric label="COLOR VARIABLE" value={colorOptions.find(([value]) => value === colorMode)?.[1] ?? colorMode} />
      </div>
      {dataset.expectedDimensionMismatch ? <p role="status" className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-small text-amber-200">Backend dimension is {records[0].dimension}, not the expected 256. The actual vectors are used as returned, without padding or truncation.</p> : null}
      {dataset.warning ? <p role="status" className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-small text-amber-200">{dataset.warning}</p> : null}
      {dataset.invalidCount > 0 ? <p role="status" className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-small text-amber-200">{dataset.invalidCount.toLocaleString()} observations excluded due to invalid or inconsistent embeddings.</p> : null}
      <p className="text-caption text-text-muted">Source / data mode: {dataset.dataMode ?? "N/A"} · Model version: {version ?? "N/A"} · Dataset version: N/A · Generated date: N/A · Embedding normalization: N/A (not reported)</p>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_370px]">
        <section className={cardClass}>
          <div className="mb-3 flex flex-wrap items-start justify-between gap-3"><div><h2 className="text-sm font-semibold uppercase tracking-wider text-text-primary">Embedding projection</h2><p className="mt-1 text-caption text-text-muted">{axisLabels[0]} × {axisLabels[1]} · one point per date response</p></div><Legend records={records} colorMode={colorMode} available={availableMeta[colorMode]} /></div>
          {projectionState.status === "computing" ? <div aria-live="polite" className="mb-2 flex justify-between text-caption text-text-secondary"><span>{projectionState.stage}</span><span className="flex items-center gap-3">{method === "tsne" ? Math.round(projectionState.progress * 100) + "%" : ""}<button type="button" onClick={() => { computationIdRef.current += 1; workerRef.current?.terminate(); workerRef.current = null; setProjectionState({ status: "cancelled", stage: "", progress: 0, error: "" }); }} className="underline underline-offset-2">Cancel computation</button></span></div> : null}
          {projectionState.status === "error" ? <p role="alert" className="mb-3 rounded-lg border border-red-500/30 p-3 text-small text-red-200">Projection unavailable: {projectionState.error}</p> : null}
          {projectionState.status === "cancelled" || projectionState.status === "error" ? <button type="button" onClick={() => setProjectionToken((value) => value + 1)} className="mb-3 rounded-lg border border-border-subtle px-3 py-2 text-caption text-text-secondary hover:bg-surface-800">Recompute projection</button> : null}
          {projectionState.status === "computing" ? <div className="mb-3 h-1 overflow-hidden rounded bg-surface-800"><div className="h-full bg-accent-400" style={{ width: Math.max(4, projectionState.progress * 100) + "%" }} /></div> : null}
          {projection ? <EmbeddingScatter points={points} mode={colorMode} axisLabels={axisLabels} selectedId={selectedId} onSelect={(id) => { setSelectedId(id); const item = records.find((record) => record.id === id); if (item) selectDate(item.date); }} /> : <div className="grid min-h-[360px] place-items-center rounded-lg border border-border-subtle bg-[#07131f] text-small text-text-muted">{projectionState.status === "error" ? "Projection computation failed." : projectionState.status === "cancelled" ? "Projection cancelled." : "Preparing embedding-space projection…"}</div>}
          <p className="mt-3 text-[11px] leading-5 text-text-muted">{method === "pca" ? "PCA is calculated from centered backend vectors; explained variance comes from its computed principal components." : "t-SNE uses seed 42 for repeatability with the same data and parameters. Its arrangement can change with settings."}</p>
          {colorMode !== "month" && !availableMeta[colorMode] ? <p role="status" className="mt-2 text-caption text-amber-200">Color metadata unavailable: /embedding does not return {colorMode === "anomaly" ? "temperature anomaly" : colorMode}.</p> : null}
        </section>
        <aside className={cardClass + " flex min-h-[400px] flex-col"} aria-label="Selected embedding details">
          <div className="mb-3 flex items-center justify-between"><div><h2 className="text-sm font-semibold uppercase tracking-wider text-text-primary">Selected embedding</h2><p className="mt-1 text-caption text-text-muted">Date-keyed backend observation</p></div>{selected && selectedPoint ? <button type="button" onClick={exportSelected} title="Export selected embedding as JSON" aria-label="Export selected embedding vector as JSON" className="rounded-lg border border-border-subtle p-2 text-text-secondary hover:bg-surface-800"><Download size={15} /></button> : null}</div>
          {selected ? <>
            <label className="mb-3 flex flex-col gap-1 text-caption text-text-muted">OBSERVATION<select aria-label="Select observation by date" className={selectClass} value={selectedId ?? ""} onChange={(event) => { setSelectedId(event.target.value); selectDate(event.target.value); }}>{records.map((record) => <option key={record.id} value={record.id}>{record.date}</option>)}</select></label>
            <dl className="grid grid-cols-2 gap-x-3 gap-y-2 border-y border-border-subtle py-3 text-caption"><Detail label="Date" value={selected.date} /><Detail label="Month" value={selectedMonth ? MONTHS[selectedMonth - 1] : "N/A"} /><Detail label="Latitude" value="N/A" /><Detail label="Longitude" value="N/A" /><Detail label="Anomaly" value="Unavailable" /><Detail label="Region" value="Unavailable" /><Detail label="Vector size" value={selected.dimension + " dimensions"} /><Detail label="PC1 explained" value={method === "pca" && projection?.explainedVariance ? (projection.explainedVariance[0] * 100).toFixed(1) + "%" : "N/A"} /><Detail label="PC2 explained" value={method === "pca" && projection?.explainedVariance ? (projection.explainedVariance[1] * 100).toFixed(1) + "%" : "N/A"} /></dl>
            <div className="mt-3 flex items-center justify-between"><button type="button" aria-expanded={vectorExpanded} onClick={() => setVectorExpanded((value) => !value)} className="text-left text-caption font-semibold uppercase tracking-wider text-text-primary">Embedding vector ({selected.dimension})</button><button type="button" onClick={() => navigator.clipboard?.writeText(JSON.stringify(selected.embedding))} className="rounded border border-border-subtle px-2 py-1 text-[11px] text-text-secondary hover:bg-surface-800">Copy vector</button></div>
            {vectorExpanded ? <div className="mt-2 max-h-[340px] overflow-auto rounded-lg border border-border-subtle bg-[#07131f] p-2" tabIndex="0" aria-label="Full embedding vector values"><table className="w-full text-right font-mono text-[11px]"><thead className="sticky top-0 bg-[#07131f] text-text-muted"><tr><th className="px-2 py-1">Dimension</th><th className="px-2 py-1">Value</th></tr></thead><tbody>{selected.embedding.map((value, index) => <tr key={index} className="border-t border-white/5"><td className="px-2 py-1 text-text-muted">[{index}]</td><td className="px-2 py-1 text-text-primary">{value.toPrecision(9)}</td></tr>)}</tbody></table></div> : <p className="mt-2 text-caption text-text-muted">Expand the table, copy, or export JSON to access all original vector values.</p>}
          </> : <p className="py-8 text-small text-text-muted">Select a plotted observation to inspect its backend vector.</p>}
        </aside>
      </div>
    </> : null}
  </div>;
}

function Metric({ label, value }) { return <div className={cardClass}><p className="text-[10px] font-semibold uppercase tracking-wider text-text-muted">{label}</p><p className="mt-1 text-base font-semibold text-text-primary">{value}</p></div>; }
function Detail({ label, value }) { return <div><dt className="text-text-muted">{label}</dt><dd className="mt-0.5 break-words font-mono text-text-primary">{value}</dd></div>; }

function Legend({ records, colorMode, available }) {
  if (colorMode !== "month" && !available) return <p className="text-caption text-text-muted">Legend · Unavailable</p>;
  if (colorMode === "month") {
    const months = [...new Set(records.map((record) => monthOfDate(record.date)).filter(Boolean))].sort((a, b) => a - b);
    return <div className="flex max-w-[440px] flex-wrap gap-x-2 gap-y-1" aria-label="Month legend">{months.map((month) => <span key={month} className="inline-flex items-center gap-1 text-[10px] text-text-secondary"><i className="h-2 w-2 rounded-full" style={{ background: "hsl(" + (((month - 1) * 30 + 185) % 360) + " 72% 56%)" }} />{MONTHS[month - 1].slice(0, 3)}</span>)}</div>;
  }
  if (colorMode === "region") {
    const regions = [...new Set(records.map((record) => record.region).filter(Boolean))].sort();
    return <div className="flex max-w-[440px] flex-wrap gap-x-2 gap-y-1" aria-label="Region legend">{regions.map((region) => <span key={region} className="inline-flex items-center gap-1 text-[10px] text-text-secondary"><i className="h-2 w-2 rounded-full" style={{ background: regionColor(region, regions) }} />{region}</span>)}</div>;
  }
  const values = records.map((record) => colorMode === "latitude" ? record.latitude : colorMode === "longitude" ? record.longitude : record.anomaly).filter((value) => Number.isFinite(value));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const extent = colorMode === "anomaly" ? Math.max(Math.abs(min), Math.abs(max)) : null;
  const label = colorMode === "anomaly" ? "Temperature anomaly" : colorMode[0].toUpperCase() + colorMode.slice(1);
  const low = colorMode === "anomaly" ? "−" + extent.toFixed(2) + " · 0" : min.toFixed(2) + (colorMode === "latitude" ? "°" : "°");
  const high = colorMode === "anomaly" ? "+" + extent.toFixed(2) : max.toFixed(2) + "°";
  return <div className="min-w-[160px] text-right text-[10px] text-text-secondary"><span className="block">{label}</span><div className="mt-1 h-2 rounded-full" style={{ background: colorMode === "anomaly" ? "linear-gradient(90deg,#3567b7,#cbd5e1,#d84b39)" : "linear-gradient(90deg,#3c91aa,#c5a13b)" }} /><span>{low} {colorMode === "anomaly" ? "→" : "→"} {high}</span></div>;
}
