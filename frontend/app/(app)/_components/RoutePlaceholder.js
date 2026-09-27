// -----------------------------------------------------------------------------
// NEER — RoutePlaceholder  (Phase 32E)
//
// Minimal stand-in rendered by each section's page.js so Sidebar/TopNavigation
// routing and active-state can be verified end to end before that section's
// real interface is built in a later phase. Intentionally has no
// visualization, data-fetching, or section-specific logic of its own — it's
// test scaffolding for the shell, not a first draft of the real page.
// -----------------------------------------------------------------------------

import { Construction } from "lucide-react";
import { Panel } from "@/components/ui";

/**
 * @param {string} title - matches the section's NAV_ITEMS label
 * @param {string} description - one line on what this section will eventually do
 */
export default function RoutePlaceholder({ title, description }) {
  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-4">
      <Panel
        title={title}
        subtitle="Not yet implemented"
        icon={Construction}
        emphasis="raised"
      >
        <p className="text-small text-text-secondary">
          {description} This route exists as a placeholder so navigation, routing, and the
          application shell can be verified — the actual interface for this section is built in a
          later phase.
        </p>
      </Panel>
    </div>
  );
}