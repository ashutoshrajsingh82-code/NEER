"use client";

// -----------------------------------------------------------------------------
// NEER Dashboard — MissionHeader  (Phase 34A, refined Phase 34B)
//
// Establishes mission identity before any live data: what NEER is, the
// program it serves (SIH26066, MoES/INCOIS), and the ocean domain + grid it
// reconstructs (region, spatial/temporal resolution). Deliberately carries no
// live operational state — system status, data mode, selected date/depth,
// and model version live in SystemStatusPanel / DataContextPanel /
// ModelContextPanel instead, so identity and live state never compete for
// the same visual weight.
//
// QA (Phase 34C): marked "use client". This component passes an icon
// component reference (`icon={RefreshCw}`) as a prop into <Button>, which is
// itself a Client Component — a Server Component doing that fails static
// generation for this route ("Functions cannot be passed directly to Client
// Components ..."). Marking this file "use client" keeps it in the same
// client subtree as Button, which resolves it without changing anything
// about how the header renders.
// -----------------------------------------------------------------------------

import { RefreshCw, Waves } from "lucide-react";
import { cn } from "@/lib/cn";
import { Badge, Button } from "@/components/ui";

const DOMAIN_FACTS = [
  { label: "Region", value: "5°N – 30°N" },
  { label: "Resolution", value: "0.25° × 0.25°" },
  { label: "Temporal", value: "Daily" },
];

export default function MissionHeader({ className }) {
  return (
    <div
      className={cn(
        "flex flex-col gap-4 border-b border-border pb-5 lg:flex-row lg:items-start lg:justify-between",
        className
      )}
    >
      <div className="flex items-start gap-4">
        <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg border border-border-accent bg-surface-raised text-accent-400 shadow-glow-sm">
          <Waves size={24} strokeWidth={1.75} aria-hidden="true" />
        </span>

        <div className="min-w-0">
          <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
            <h1 className="text-h2 font-semibold tracking-tight text-text-primary">NEER</h1>
            <span className="text-small text-text-muted">Neural Estimation of Essential ocean Records</span>
          </div>
          <p className="mt-1 text-caption text-text-secondary">Satellite-to-subsurface temperature reconstruction · North Indian Ocean</p>

          <div className="mt-2.5 flex flex-wrap items-center gap-2">
            <Badge variant="accent">SIH26066</Badge>
            <Badge variant="default">MoES / INCOIS</Badge>
            <Badge variant="neutral">North Indian Ocean</Badge>
          </div>

          {/* Scientific scope — a compact instrument-plate readout rather than
              another row of cards/badges. */}
          <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-caption">
            {DOMAIN_FACTS.map((fact, index) => (
              <span key={fact.label} className="flex items-center gap-1.5">
                {index > 0 && <span className="h-3 w-px bg-border" aria-hidden="true" />}
                <span className="text-text-disabled">{fact.label.toUpperCase()}</span>
                <span className="text-text-secondary">{fact.value}</span>
              </span>
            ))}
          </div>
        </div>
      </div>

      <Button variant="secondary" size="sm" icon={RefreshCw} className="shrink-0 lg:mt-1.5">
        Refresh
      </Button>
    </div>
  );
}
