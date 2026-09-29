"use client";

import { useApiRequest } from "@/lib/useApiRequest";
import { health } from "@/lib/api";
import { useNeerContext } from "../_context/NeerContext";
import SystemStatusPanel from "./SystemStatusPanel";
import ModelContextPanel from "./ModelContextPanel";

export default function DashboardRuntimePanel() {
  const { data, isLoading, error } = useApiRequest(health, { immediate: true });
  const { modelInfoData } = useNeerContext();
  const state = isLoading ? "processing" : error ? "degraded" : data?.status === "ok" ? "operational" : "degraded";
  const detail = isLoading ? "Checking backend components…" : error ? error.message :
    `API ${data?.components?.api?.status ?? "unknown"} · data ${data?.components?.data?.status ?? "unknown"} · model ${data?.components?.model?.status ?? "unknown"}`;
  const checkpoint = data?.components?.model?.checkpoint;
  const version = checkpoint || modelInfoData?.checkpoint?.filename || "Checkpoint unavailable";
  const synthetic = data?.components?.data?.is_synthetic;
  return <>
    <SystemStatusPanel state={state} detail={detail} />
    <div className="flex flex-col gap-1 sm:border-l sm:border-border-subtle sm:pl-6">
      <ModelContextPanel version={version} name="Loaded NEER checkpoint" />
      <p className="text-caption text-text-muted">Environment: {modelInfoData?.environment ?? "Unavailable"} · Dataset: {synthetic === true ? "DEMO_SYNTHETIC (synthetic; not independent validation)" : synthetic === false ? "non-synthetic" : "Unavailable"}</p>
    </div>
  </>;
}
