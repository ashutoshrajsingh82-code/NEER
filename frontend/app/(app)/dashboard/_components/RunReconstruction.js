"use client";

import { useRef, useState } from "react";
import { Play } from "lucide-react";
import { Button, MetricCard } from "@/components/ui";
import { runReconstruction } from "@/lib/api";
import { OCEAN_DOMAIN, isWithinDomain } from "@/lib/oceanDomain";
import { formatDate, formatSigned } from "@/lib/format";
import { isDateInDataRange } from "@/lib/dateDepthModel";

/** Explicit, measured point reconstruction using the selected dashboard context. */
export default function RunReconstruction({ date, depth, availableDates, availableDepths, selectedPoint }) {
  const activeRef = useRef(false);
  const [request, setRequest] = useState({ status: "idle", result: null, error: null, latencyMs: null });
  const dateValid = typeof date === "string" && isDateInDataRange(date, availableDates);
  const depthValid = availableDepths.includes(depth);
  const pointValid = selectedPoint && isWithinDomain(selectedPoint.lat, selectedPoint.lon, OCEAN_DOMAIN);
  const canRun = dateValid && depthValid && pointValid;

  async function run() {
    if (activeRef.current) return;
    if (!dateValid || !depthValid || !pointValid) {
      const error = !dateValid
        ? "Select a date inside the date range covered by the NEER backend."
        : !depthValid
          ? "Selected depth is not supported by the NEER model."
          : "Select an ocean coordinate inside the NEER domain on the map.";
      setRequest((current) => ({ ...current, status: "error", error }));
      return;
    }

    activeRef.current = true;
    const started = performance.now();
    setRequest((current) => ({ ...current, status: "running", error: null }));
    try {
      const result = await runReconstruction({
        date,
        depth,
        latitude: selectedPoint.lat,
        longitude: selectedPoint.lon,
      });
      setRequest({ status: "success", result, error: null, latencyMs: performance.now() - started });
    } catch (error) {
      setRequest((current) => ({ ...current, status: "error", error: error?.message || "Reconstruction request failed." }));
    } finally {
      activeRef.current = false;
    }
  }

  const oldResult = request.result && (
    request.result.date !== date || request.result.depth !== depth ||
    request.result.lat !== selectedPoint?.lat || request.result.lon !== selectedPoint?.lon
  );

  return (
    <section className="rounded-lg border border-border bg-surface-raised p-4" aria-label="Run reconstruction">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-small font-semibold text-text-primary">Live reconstruction</h2>
          <p className="mt-1 text-caption text-text-muted">
            {selectedPoint
              ? `${formatDate(date)} · ${depth ?? "--"} m · ${selectedPoint.lat.toFixed(2)}°N, ${selectedPoint.lon.toFixed(2)}°E`
              : "Select an ocean point on the map, then run the backend reconstruction."}
          </p>
        </div>
        <Button
          variant="primary"
          icon={Play}
          loading={request.status === "running"}
          disabled={request.status === "running"}
          onClick={run}
        >
          {request.status === "running" ? "RECONSTRUCTING…" : "RUN RECONSTRUCTION"}
        </Button>
      </div>

      {request.status === "running" && (
        <ol className="mt-3 border-t border-border-subtle pt-3 text-caption text-text-secondary" aria-live="polite">
          <li>Request sent to NEER backend · {formatDate(date)} · {depth} m</li>
          <li className="mt-1">Waiting for validated reconstruction response</li>
        </ol>
      )}

      {request.status === "error" && (
        <p className="mt-3 border-t border-border-subtle pt-3 text-caption text-error" role="alert">
          Reconstruction failed: {request.error}
          {request.result && <span className="ml-1 text-text-muted">The last successful result remains available above only for its original context.</span>}
        </p>
      )}

      {request.result && (
        <div className="mt-3 border-t border-border-subtle pt-3" aria-live="polite">
          {oldResult && <p className="mb-2 text-caption text-warning">Previous successful result · {formatDate(request.result.date)} · {request.result.depth} m</p>}
          {!oldResult && request.status !== "success" && <p className="mb-2 text-caption text-text-muted">Last successful response · {formatDate(request.result.date)} · {request.result.depth} m</p>}
          {!oldResult && request.status === "success" && <p className="mb-2 text-caption text-accent-300">Reconstruction complete · {formatDate(request.result.date)} · {request.result.depth} m · {request.latencyMs.toFixed(0)} ms measured request latency</p>}
          <div className="grid gap-3 sm:grid-cols-2">
            <MetricCard title="MODEL OUTPUT" value={typeof request.result.temperature === "number" && Number.isFinite(request.result.temperature) ? request.result.temperature.toFixed(2) : "N/A"} unit={typeof request.result.temperature === "number" && Number.isFinite(request.result.temperature) ? "°C" : undefined} />
            <MetricCard title="ANOMALY" value={typeof request.result.anomaly === "number" && Number.isFinite(request.result.anomaly) ? formatSigned(request.result.anomaly) : "N/A"} unit={typeof request.result.anomaly === "number" && Number.isFinite(request.result.anomaly) ? "°C" : undefined} />
          </div>
        </div>
      )}
    </section>
  );
}
