"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { continuousColor, getEmbeddingColor, monthColor, regionColor } from "@/lib/embeddingExplorer";

const MARGIN = { top: 24, right: 24, bottom: 48, left: 58 };

function pointColor(record, mode, domain, categories) {
  const value = getEmbeddingColor(record, mode);
  if (value === null) return "#64748b";
  if (mode === "month") return monthColor(value);
  if (mode === "region") return regionColor(value, categories);
  return continuousColor(value, domain, mode);
}

export default function EmbeddingScatter({ points, mode, axisLabels, selectedId, onSelect }) {
  const canvasRef = useRef(null);
  const [size, setSize] = useState({ width: 720, height: 480 });
  const [hovered, setHovered] = useState(null);
  const extent = useMemo(() => {
    if (!points.length) return null;
    let xMin = Infinity; let xMax = -Infinity; let yMin = Infinity; let yMax = -Infinity;
    for (const point of points) {
      xMin = Math.min(xMin, point.x); xMax = Math.max(xMax, point.x);
      yMin = Math.min(yMin, point.y); yMax = Math.max(yMax, point.y);
    }
    const xPad = (xMax - xMin) * 0.06 || 1;
    const yPad = (yMax - yMin) * 0.06 || 1;
    return { xMin: xMin - xPad, xMax: xMax + xPad, yMin: yMin - yPad, yMax: yMax + yPad };
  }, [points]);
  const domain = useMemo(() => {
    const values = points.map((point) => getEmbeddingColor(point.record, mode)).filter((value) => value !== null);
    if (mode === "month" || mode === "region") return [...new Set(values)].sort((a, b) => typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b)));
    if (!values.length) return null;
    const min = Math.min(...values); const max = Math.max(...values);
    return mode === "anomaly" ? [-Math.max(Math.abs(min), Math.abs(max)), Math.max(Math.abs(min), Math.abs(max))] : [min, max];
  }, [points, mode]);

  const plotPoint = useCallback((point) => {
    if (!extent) return null;
    const plotW = size.width - MARGIN.left - MARGIN.right;
    const plotH = size.height - MARGIN.top - MARGIN.bottom;
    return {
      x: MARGIN.left + ((point.x - extent.xMin) / (extent.xMax - extent.xMin)) * plotW,
      y: MARGIN.top + (1 - (point.y - extent.yMin) / (extent.yMax - extent.yMin)) * plotH,
    };
  }, [extent, size]);
  const screenPoints = useMemo(() => points.map((point) => ({ ...plotPoint(point), point })), [points, plotPoint]);
  const hitGrid = useMemo(() => {
    const grid = new Map();
    for (const item of screenPoints) {
      const key = Math.floor(item.x / 12) + ":" + Math.floor(item.y / 12);
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key).push(item);
    }
    return grid;
  }, [screenPoints]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const observer = new ResizeObserver(([entry]) => {
      setSize({ width: Math.max(320, entry.contentRect.width), height: Math.max(340, Math.min(620, entry.contentRect.width * 0.64)) });
    });
    observer.observe(canvas.parentElement);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !extent) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(size.width * ratio);
    canvas.height = Math.round(size.height * ratio);
    canvas.style.width = `${size.width}px`;
    canvas.style.height = `${size.height}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, size.width, size.height);
    const plotW = size.width - MARGIN.left - MARGIN.right;
    const plotH = size.height - MARGIN.top - MARGIN.bottom;
    ctx.fillStyle = "#07131f";
    ctx.fillRect(MARGIN.left, MARGIN.top, plotW, plotH);
    ctx.strokeStyle = "rgba(148, 163, 184, .13)";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#71849a";
    ctx.font = "11px ui-monospace, monospace";
    for (let i = 0; i <= 4; i += 1) {
      const x = MARGIN.left + (plotW * i) / 4;
      const y = MARGIN.top + (plotH * i) / 4;
      ctx.beginPath(); ctx.moveTo(x, MARGIN.top); ctx.lineTo(x, MARGIN.top + plotH); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(MARGIN.left, y); ctx.lineTo(MARGIN.left + plotW, y); ctx.stroke();
      const xv = extent.xMin + ((extent.xMax - extent.xMin) * i) / 4;
      const yv = extent.yMax - ((extent.yMax - extent.yMin) * i) / 4;
      ctx.fillText(xv.toPrecision(3), x - 14, MARGIN.top + plotH + 18);
      ctx.fillText(yv.toPrecision(3), 6, y + 4);
    }
    ctx.strokeStyle = "rgba(148,163,184,.48)";
    ctx.beginPath(); ctx.moveTo(MARGIN.left, MARGIN.top + plotH); ctx.lineTo(MARGIN.left + plotW, MARGIN.top + plotH); ctx.lineTo(MARGIN.left + plotW, MARGIN.top); ctx.stroke();
    for (const item of screenPoints) {
      const point = item.point;
      const xy = item;
      const selected = point.record.id === selectedId;
      const radius = selected ? 6 : screenPoints.length > 4000 ? 1.8 : 3;
      ctx.beginPath(); ctx.arc(xy.x, xy.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = pointColor(point.record, mode, domain, domain || []);
      ctx.globalAlpha = 0.84;
      ctx.fill(); ctx.globalAlpha = 1;
      if (selected) { ctx.strokeStyle = "#f8fafc"; ctx.lineWidth = 2; ctx.stroke(); }
    }
    ctx.fillStyle = "#a9bdcf";
    ctx.font = "12px ui-sans-serif, system-ui";
    ctx.textAlign = "center";
    ctx.fillText(axisLabels[0], MARGIN.left + plotW / 2, size.height - 8);
    ctx.save();
    ctx.translate(15, MARGIN.top + plotH / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText(axisLabels[1], 0, 0);
    ctx.restore();
  }, [screenPoints, mode, domain, extent, size, selectedId, axisLabels]);

  const findPoint = (event) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (size.width / rect.width);
    const y = (event.clientY - rect.top) * (size.height / rect.height);
    let closest = null; let distance = 11;
    const cellX = Math.floor(x / 12); const cellY = Math.floor(y / 12);
    for (let dx = -1; dx <= 1; dx += 1) for (let dy = -1; dy <= 1; dy += 1) {
      const candidates = hitGrid.get((cellX + dx) + ":" + (cellY + dy)) ?? [];
      for (const item of candidates) {
        const d = Math.hypot(item.x - x, item.y - y);
        if (d < distance) { closest = item.point; distance = d; }
      }
    }
    return closest;
  };
  const handleMove = (event) => {
    const point = findPoint(event);
    setHovered(point?.record.id ?? null);
  };
  const active = hovered ? points.find((point) => point.record.id === hovered)?.record : null;

  return (
    <div className="relative w-full overflow-hidden rounded-xl border border-border-subtle bg-[#07131f]" onPointerLeave={() => setHovered(null)}>
      <canvas ref={canvasRef} role="img" aria-label={`Embedding scatter plot. Axes ${axisLabels[0]} and ${axisLabels[1]}. Use the observation selector or click a point to inspect it.`} onPointerMove={handleMove} onClick={(event) => { const point = findPoint(event); if (point) onSelect(point.record.id); }} className="block max-w-full cursor-crosshair" />
      {active ? <div className="pointer-events-none absolute left-3 top-3 rounded-lg border border-border-subtle bg-[#0b1927]/95 px-3 py-2 text-xs shadow-xl">
        <div className="font-mono text-text-primary">{active.date}</div>
        <div className="mt-1 text-text-muted">Latitude: N/A · Longitude: N/A</div>
        <div className="text-text-muted">Anomaly: Unavailable · Region: Unavailable</div>
      </div> : null}
    </div>
  );
}
