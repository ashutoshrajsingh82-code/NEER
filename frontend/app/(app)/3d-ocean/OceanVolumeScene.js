"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { volumeColor, volumeValue } from "@/lib/oceanVolume";
import { OCEAN_DOMAIN } from "@/lib/oceanDomain";

const VERTEX_SHADER = `attribute vec3 a_position; attribute vec3 a_color; attribute vec4 a_pick; attribute float a_selected; uniform mat4 u_mvp; uniform float u_size; uniform float u_pick_size; uniform bool u_pick_mode; varying vec3 v_color; varying vec4 v_pick; varying float v_selected; void main(){ gl_Position=u_mvp*vec4(a_position,1.0); gl_PointSize=u_pick_mode?u_pick_size:u_size*(a_selected>0.5?2.6:1.0); v_color=a_color; v_pick=a_pick; v_selected=a_selected; }`;
const FRAGMENT_SHADER = `precision highp float; uniform bool u_pick_mode; varying vec3 v_color; varying vec4 v_pick; varying float v_selected; void main(){ vec2 q=gl_PointCoord-vec2(0.5); float r=length(q); if(r>0.5) discard; if(u_pick_mode){ gl_FragColor=v_pick; return; } if(v_selected>0.5 && r>0.34){ gl_FragColor=vec4(1.0,0.88,0.35,1.0); } else { gl_FragColor=vec4(v_color,1.0); } }`;

function compile(gl, type, source) {
  const shader = gl.createShader(type); gl.shaderSource(shader, source); gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) { const message = gl.getShaderInfoLog(shader); gl.deleteShader(shader); throw new Error(message || "WebGL shader compilation failed."); }
  return shader;
}
function mul(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c += 1) for (let r = 0; r < 4; r += 1) out[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  return out;
}
function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2); const out = new Float32Array(16);
  out[0] = f / aspect; out[5] = f; out[10] = (far + near) / (near - far); out[11] = -1; out[14] = (2 * far * near) / (near - far); return out;
}
function lookAt(eye, center, up) {
  const norm = (v) => { const n = Math.hypot(...v) || 1; return v.map((x) => x / n); };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const sub = (a, b) => a.map((v, i) => v - b[i]);
  const z = norm(sub(eye, center)); const x = norm(cross(up, z)); const y = cross(z, x); const out = new Float32Array(16);
  out[0] = x[0]; out[1] = y[0]; out[2] = z[0]; out[3] = 0;
  out[4] = x[1]; out[5] = y[1]; out[6] = z[1]; out[7] = 0;
  out[8] = x[2]; out[9] = y[2]; out[10] = z[2]; out[11] = 0;
  out[12] = -x.reduce((s, v, i) => s + v * eye[i], 0); out[13] = -y.reduce((s, v, i) => s + v * eye[i], 0); out[14] = -z.reduce((s, v, i) => s + v * eye[i], 0); out[15] = 1; return out;
}

function makePoints(volume, variable, minDepth, maxDepth, selectedPoint, selectedDepth) {
  let count = 0;
  volume.depths.forEach((depth, k) => { if (depth >= minDepth && depth <= maxDepth) count += volume.lat.length * volume.lon.length; });
  const positions = new Float32Array(count * 3); const colors = new Uint8Array(count * 3); const picks = new Uint8Array(count * 4); const selected = new Float32Array(count);
  const latIndex = new Uint16Array(count); const lonIndex = new Uint16Array(count); const depthIndex = new Uint8Array(count); const values = new Float32Array(count); const missing = new Uint8Array(count);
  const realValues = [];
  let p = 0;
  volume.depths.forEach((depth, k) => {
    if (depth < minDepth || depth > maxDepth) return;
    for (let i = 0; i < volume.lat.length; i += 1) for (let j = 0; j < volume.lon.length; j += 1) {
      const value = volumeValue(volume, i, j, k, variable); if (typeof value === "number" && Number.isFinite(value)) realValues.push(value);
      positions[p * 3] = ((volume.lon[j] - volume.lon[0]) / Math.max(1e-9, volume.lon.at(-1) - volume.lon[0])) * 2 - 1;
      positions[p * 3 + 1] = 0.82 - (depth / Math.max(...volume.depths)) * 1.64;
      positions[p * 3 + 2] = ((volume.lat[i] - volume.lat[0]) / Math.max(1e-9, volume.lat.at(-1) - volume.lat[0])) * 1.55 - 0.775;
      latIndex[p] = i; lonIndex[p] = j; depthIndex[p] = k; values[p] = typeof value === "number" ? value : NaN; missing[p] = typeof value === "number" ? 0 : 1;
      const id = p + 1; picks[p * 4] = id & 255; picks[p * 4 + 1] = (id >> 8) & 255; picks[p * 4 + 2] = (id >> 16) & 255; picks[p * 4 + 3] = 255;
      selected[p] = selectedPoint && Math.abs(volume.lat[i] - selectedPoint.lat) < OCEAN_DOMAIN.resolution / 2 && Math.abs(volume.lon[j] - selectedPoint.lon) < OCEAN_DOMAIN.resolution / 2 && depth === selectedDepth ? 1 : 0;
      p += 1;
    }
  });
  const lo = realValues.length ? realValues.reduce((best, value) => Math.min(best, value), Infinity) : 0; const hi = realValues.length ? realValues.reduce((best, value) => Math.max(best, value), -Infinity) : 1;
  for (let i = 0; i < count; i += 1) {
    const rgb = missing[i] ? [92, 104, 117] : volumeColor(values[i], lo, hi, variable);
    colors.set(rgb, i * 3);
  }
  return { positions, colors, picks, selected, latIndex, lonIndex, depthIndex, values, missing, count, min: lo, max: hi, hasValues: realValues.length > 0 };
}

export default function OceanVolumeScene({ volume, variable, minDepth, maxDepth, selectedDepth, selectedPoint, onInspect, onSelect, resetToken }) {
  const canvasRef = useRef(null); const stateRef = useRef({ yaw: 0.72, pitch: 0.48, distance: 3.1, panX: 0, panZ: 0 }); const redrawRef = useRef(null);
  const [unavailable, setUnavailable] = useState(false); const [rendered, setRendered] = useState({ min: 0, max: 1, count: 0 });
  const points = useMemo(() => makePoints(volume, variable, minDepth, maxDepth, selectedPoint, selectedDepth), [volume, variable, minDepth, maxDepth, selectedPoint, selectedDepth]);

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return undefined;
    let gl; let program; let positionBuffer; let colorBuffer; let pickBuffer; let selectedBuffer; let pickFramebuffer; let pickTexture; let pickDepth;
    let resizeObserver; let active = true; let pointerFrame = 0; let drag = null; let pinchDistance = null; const touchPoints = new Map(); let lastPick = -1;
    try {
      gl = canvas.getContext("webgl", { antialias: true, alpha: false, preserveDrawingBuffer: false });
      if (!gl) { setUnavailable(true); return undefined; }
      setUnavailable(false);
      const vs = compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER); const fs = compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
      program = gl.createProgram(); gl.attachShader(program, vs); gl.attachShader(program, fs); gl.linkProgram(program);
      gl.deleteShader(vs); gl.deleteShader(fs);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) || "WebGL program link failed.");
      gl.useProgram(program);
      positionBuffer = gl.createBuffer(); colorBuffer = gl.createBuffer(); pickBuffer = gl.createBuffer(); selectedBuffer = gl.createBuffer();
      const aPosition = gl.getAttribLocation(program, "a_position"); const aColor = gl.getAttribLocation(program, "a_color"); const aPick = gl.getAttribLocation(program, "a_pick"); const aSelected = gl.getAttribLocation(program, "a_selected");
      const uMvp = gl.getUniformLocation(program, "u_mvp"); const uSize = gl.getUniformLocation(program, "u_size"); const uPickSize = gl.getUniformLocation(program, "u_pick_size"); const uPickMode = gl.getUniformLocation(program, "u_pick_mode");
      gl.enable(gl.DEPTH_TEST);
      const bindAttrib = (buffer, location, size, type, normalized) => { gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, size, type, normalized, 0, 0); };
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer); gl.bufferData(gl.ARRAY_BUFFER, points.positions, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer); gl.bufferData(gl.ARRAY_BUFFER, points.colors, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, pickBuffer); gl.bufferData(gl.ARRAY_BUFFER, points.picks, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, selectedBuffer); gl.bufferData(gl.ARRAY_BUFFER, points.selected, gl.STATIC_DRAW);
      setRendered({ min: points.min, max: points.max, count: points.count, hasValues: points.hasValues });
      const ensurePicker = () => {
        if (pickFramebuffer) { gl.deleteFramebuffer(pickFramebuffer); gl.deleteTexture(pickTexture); gl.deleteRenderbuffer(pickDepth); }
        pickFramebuffer = gl.createFramebuffer(); pickTexture = gl.createTexture(); pickDepth = gl.createRenderbuffer();
        gl.bindTexture(gl.TEXTURE_2D, pickTexture); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, canvas.width, canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.bindRenderbuffer(gl.RENDERBUFFER, pickDepth); gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, canvas.width, canvas.height);
        gl.bindFramebuffer(gl.FRAMEBUFFER, pickFramebuffer); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, pickTexture, 0); gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, pickDepth); gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      };
      const render = () => {
        if (!active) return;
        const rect = canvas.getBoundingClientRect(); const dpr = Math.min(1.5, window.devicePixelRatio || 1);
        const w = Math.max(1, Math.floor(rect.width * dpr)); const h = Math.max(1, Math.floor(rect.height * dpr));
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; ensurePicker(); }
        gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, w, h); gl.clearColor(0.025, 0.07, 0.12, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        const { yaw, pitch, distance, panX, panZ } = stateRef.current;
        const target = [panX, 0, panZ]; const eye = [target[0] + distance * Math.sin(yaw) * Math.cos(pitch), target[1] + distance * Math.sin(pitch), target[2] + distance * Math.cos(yaw) * Math.cos(pitch)];
        const mvp = mul(perspective(Math.PI / 4, w / h, 0.05, 30), lookAt(eye, target, [0, 1, 0]));
        gl.useProgram(program); gl.uniformMatrix4fv(uMvp, false, mvp); gl.uniform1f(uSize, Math.max(2, Math.min(5, 3 * dpr))); gl.uniform1f(uPickSize, 12 * dpr); gl.uniform1i(uPickMode, 0);
        bindAttrib(positionBuffer, aPosition, 3, gl.FLOAT, false); bindAttrib(colorBuffer, aColor, 3, gl.UNSIGNED_BYTE, true); bindAttrib(pickBuffer, aPick, 4, gl.UNSIGNED_BYTE, true); bindAttrib(selectedBuffer, aSelected, 1, gl.FLOAT, false);
        gl.drawArrays(gl.POINTS, 0, points.count);
      };
      redrawRef.current = render;
      const pickAt = (clientX, clientY) => {
        const rect = canvas.getBoundingClientRect(); const dpr = canvas.width / Math.max(1, rect.width);
        const x = Math.max(0, Math.min(canvas.width - 1, Math.floor((clientX - rect.left) * dpr)));
        const y = Math.max(0, Math.min(canvas.height - 1, canvas.height - 1 - Math.floor((clientY - rect.top) * dpr)));
        render(); gl.bindFramebuffer(gl.FRAMEBUFFER, pickFramebuffer); gl.viewport(0, 0, canvas.width, canvas.height); gl.enable(gl.SCISSOR_TEST); gl.scissor(Math.max(0, x - 3), Math.max(0, y - 3), Math.min(7, canvas.width), Math.min(7, canvas.height)); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.useProgram(program); gl.uniform1i(uPickMode, 1); gl.uniform1f(uPickSize, 12 * dpr); const { yaw, pitch, distance, panX, panZ } = stateRef.current; const target = [panX, 0, panZ]; const eye = [target[0] + distance * Math.sin(yaw) * Math.cos(pitch), target[1] + distance * Math.sin(pitch), target[2] + distance * Math.cos(yaw) * Math.cos(pitch)]; gl.uniformMatrix4fv(uMvp, false, mul(perspective(Math.PI / 4, canvas.width / canvas.height, 0.05, 30), lookAt(eye, target, [0, 1, 0]))); gl.drawArrays(gl.POINTS, 0, points.count);
        const pixel = new Uint8Array(4); gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel); gl.disable(gl.SCISSOR_TEST); gl.bindFramebuffer(gl.FRAMEBUFFER, null); render();
        const id = (pixel[0] | (pixel[1] << 8) | (pixel[2] << 16)) - 1;
        if (id < 0 || id >= points.count) { lastPick = -1; onInspect(null); return null; }
        if (id !== lastPick) {
          lastPick = id;
          onInspect({ lat: volume.lat[points.latIndex[id]], lon: volume.lon[points.lonIndex[id]], depth: volume.depths[points.depthIndex[id]], value: points.missing[id] ? null : points.values[id], variable, dataMode: volume.data_mode, date: volume.date });
        }
        return { id, lat: volume.lat[points.latIndex[id]], lon: volume.lon[points.lonIndex[id]], depth: volume.depths[points.depthIndex[id]] };
      };
      const down = (event) => {
        if (event.button !== 0) return;
        canvas.setPointerCapture(event.pointerId);
        if (event.pointerType === "touch") {
          touchPoints.set(event.pointerId, [event.clientX, event.clientY]);
          if (touchPoints.size === 2) { const pts = [...touchPoints.values()]; pinchDistance = Math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]); drag = null; return; }
        }
        drag = { x: event.clientX, y: event.clientY, moved: false, shift: event.shiftKey };
      };
      const move = (event) => {
        if (event.pointerType === "touch" && touchPoints.has(event.pointerId)) {
          touchPoints.set(event.pointerId, [event.clientX, event.clientY]);
          if (touchPoints.size >= 2) { const pts = [...touchPoints.values()]; const nextDistance = Math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]); if (pinchDistance && nextDistance > 0) stateRef.current.distance = Math.max(1.25, Math.min(8, stateRef.current.distance * pinchDistance / nextDistance)); pinchDistance = nextDistance; drag = null; render(); return; }
        }
        if (drag) {
          const dx = event.clientX - drag.x; const dy = event.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
          if (drag.moved) { if (drag.shift) { stateRef.current.panX = Math.max(-0.8, Math.min(0.8, stateRef.current.panX - dx * 0.003)); stateRef.current.panZ = Math.max(-0.8, Math.min(0.8, stateRef.current.panZ + dy * 0.003)); } else { stateRef.current.yaw += dx * 0.008; stateRef.current.pitch = Math.max(-1.25, Math.min(1.25, stateRef.current.pitch + dy * 0.008)); } render(); }
          drag.x = event.clientX; drag.y = event.clientY;
        } else if (!pointerFrame) pointerFrame = requestAnimationFrame(() => { pointerFrame = 0; pickAt(event.clientX, event.clientY); });
      };
      const up = (event) => {
        if (event.pointerType === "touch") { touchPoints.delete(event.pointerId); if (touchPoints.size < 2) pinchDistance = null; if (!touchPoints.size) drag = null; }
        if (!drag) return;
        const wasMoved = drag.moved; drag = null; if (!wasMoved) { const point = pickAt(event.clientX, event.clientY); if (point) onSelect(point); }
      };
      const wheel = (event) => { event.preventDefault(); stateRef.current.distance = Math.max(1.25, Math.min(8, stateRef.current.distance * (event.deltaY > 0 ? 1.08 : 0.92))); render(); };
      const keydown = (event) => {
        const step = 0.08;
        if (event.key === "ArrowLeft") stateRef.current.yaw -= step;
        else if (event.key === "ArrowRight") stateRef.current.yaw += step;
        else if (event.key === "ArrowUp") stateRef.current.pitch = Math.min(1.25, stateRef.current.pitch + step);
        else if (event.key === "ArrowDown") stateRef.current.pitch = Math.max(-1.25, stateRef.current.pitch - step);
        else if (event.key === "+" || event.key === "=") stateRef.current.distance = Math.max(1.25, stateRef.current.distance * 0.9);
        else if (event.key === "-") stateRef.current.distance = Math.min(8, stateRef.current.distance * 1.1);
        else if (event.key === "Enter") { const rect = canvas.getBoundingClientRect(); const point = pickAt(rect.left + rect.width / 2, rect.top + rect.height / 2); if (point) onSelect(point); return; }
        else return;
        event.preventDefault(); render();
      };
      const leave = () => { if (!drag) { lastPick = -1; onInspect(null); } };
      canvas.tabIndex = 0;
      canvas.addEventListener("pointerdown", down); canvas.addEventListener("pointermove", move); canvas.addEventListener("pointerup", up); canvas.addEventListener("pointercancel", up); canvas.addEventListener("pointerleave", leave); canvas.addEventListener("wheel", wheel, { passive: false }); canvas.addEventListener("keydown", keydown);
      resizeObserver = new ResizeObserver(render); resizeObserver.observe(canvas); render();
      canvas.__oceanClean = () => { canvas.removeEventListener("pointerdown", down); canvas.removeEventListener("pointermove", move); canvas.removeEventListener("pointerup", up); canvas.removeEventListener("pointercancel", up); canvas.removeEventListener("pointerleave", leave); canvas.removeEventListener("wheel", wheel); canvas.removeEventListener("keydown", keydown); };
      return () => { active = false; resizeObserver?.disconnect(); if (pointerFrame) cancelAnimationFrame(pointerFrame); canvas.__oceanClean?.(); gl.deleteBuffer(positionBuffer); gl.deleteBuffer(colorBuffer); gl.deleteBuffer(pickBuffer); gl.deleteBuffer(selectedBuffer); gl.deleteProgram(program); if (pickFramebuffer) gl.deleteFramebuffer(pickFramebuffer); if (pickTexture) gl.deleteTexture(pickTexture); if (pickDepth) gl.deleteRenderbuffer(pickDepth); redrawRef.current = null; };
    } catch {
      setUnavailable(true);
      return () => { active = false; resizeObserver?.disconnect(); if (gl) { if (positionBuffer) gl.deleteBuffer(positionBuffer); if (colorBuffer) gl.deleteBuffer(colorBuffer); if (pickBuffer) gl.deleteBuffer(pickBuffer); if (selectedBuffer) gl.deleteBuffer(selectedBuffer); if (program) gl.deleteProgram(program); } };
    }
  }, [volume, points, variable, onInspect, onSelect]);

  useEffect(() => { if (resetToken > 0) { stateRef.current = { yaw: 0.72, pitch: 0.48, distance: 3.1, panX: 0, panZ: 0 }; redrawRef.current?.(); } }, [resetToken]);
  const anomalyExtent = Math.max(Math.abs(rendered.min), Math.abs(rendered.max));
  const legend = !rendered.hasValues ? "No finite values" : variable === "anomaly" ? `−${anomalyExtent.toFixed(2)} to +${anomalyExtent.toFixed(2)} °C` : `${rendered.min.toFixed(2)} to ${rendered.max.toFixed(2)} °C`;
  return <div className="relative overflow-hidden rounded-lg border border-border-subtle bg-[#06121e]">
    {unavailable ? <div role="alert" className="flex min-h-[28rem] items-center justify-center p-8 text-center text-small text-text-secondary">3D visualization is unavailable in this browser or graphics environment. Use the <a className="ml-1 text-accent-300 underline" href="/dashboard">2D Ocean Map</a> or <a className="ml-1 text-accent-300 underline" href="/vertical-profile">Vertical Profile</a>.</div> : <>
      <canvas ref={canvasRef} aria-label="Interactive 3D ocean volume. Drag to rotate, shift-drag to pan, scroll or pinch to zoom, hover for model values, and click a point to select its coordinates and depth. When focused, use arrow keys to rotate, plus or minus to zoom, and Enter to select the center point." className="block h-[min(72vh,42rem)] min-h-[28rem] w-full touch-none cursor-grab outline-none focus-visible:ring-2 focus-visible:ring-accent-300 active:cursor-grabbing" />
      <div className="pointer-events-none absolute left-3 top-3 rounded bg-slate-950/75 px-3 py-2 text-[11px] text-slate-200">Longitude → &nbsp; Latitude ↗ &nbsp; Depth ↓</div>
      <div className="pointer-events-none absolute bottom-3 left-3 rounded bg-slate-950/75 px-3 py-2 text-[11px] text-slate-200">{rendered.count.toLocaleString()} model grid points · {variable === "temperature" ? "MODEL OUTPUT · Temperature (°C)" : `ANOMALY · negative ← 0 → positive (°C)`}: {legend} · Gray points: missing</div>
    </>}
  </div>;
}
