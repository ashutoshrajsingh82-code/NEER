// -----------------------------------------------------------------------------
// NEER Dashboard — PointInspectionPlaceholder  (Phase 34A)
//
// Content rendered inside the shared shell InspectionPanel while the
// dashboard route is active and no map point has been selected yet. Shows
// the shape of what a real selection will surface (a later phase) using
// dashed placeholder rows, rather than a bare empty state.
// -----------------------------------------------------------------------------

import { MapPin } from "lucide-react";
import { Badge, Panel } from "@/components/ui";

const PREVIEW_FIELDS = [
  { label: "Sea surface temperature", unit: "°C" },
  { label: "Sea surface salinity", unit: "PSU" },
  { label: "Depth", unit: "m" },
  { label: "Reconstruction confidence", unit: "%" },
];

export default function PointInspectionPlaceholder() {
  return (
    <div className="flex flex-col gap-4">
      <Panel emphasis="base" icon={MapPin} title="No point selected" bodyClassName="flex flex-col gap-3">
        <p className="text-small text-text-secondary">
          Click a location on the ocean map to inspect its reconstructed field values here.
        </p>
        <dl className="flex flex-col gap-2">
          {PREVIEW_FIELDS.map((field) => (
            <div
              key={field.label}
              className="flex items-center justify-between rounded-md border border-dashed border-border px-3 py-2"
            >
              <dt className="text-caption text-text-muted">{field.label}</dt>
              <dd className="font-mono text-small text-text-disabled">-- {field.unit}</dd>
            </div>
          ))}
        </dl>
      </Panel>

      <Badge variant="neutral" size="sm" className="self-start">
        Grid: 5°N–30°N · 0.25° resolution
      </Badge>
    </div>
  );
}