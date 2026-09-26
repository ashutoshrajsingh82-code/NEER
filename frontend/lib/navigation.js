// -----------------------------------------------------------------------------
// NEER Navigation — configuration
//
// The single source of truth for NEER's primary navigation. Kept separate
// from <Sidebar> so the same list can drive the sidebar, the top bar's
// current-section indicator, and (later) things like a command palette or
// breadcrumbs, without duplicating the item list.
//
// Routes are placeholders — the destination pages are built in later phases.
// -----------------------------------------------------------------------------

import {
  Activity,
  AreaChart,
  Box,
  ClipboardCheck,
  LayoutDashboard,
  Lightbulb,
  MoveVertical,
  Network,
  ShieldCheck,
  Waves,
} from "lucide-react";

/**
 * @typedef {Object} NavItem
 * @property {string} value - stable identifier
 * @property {string} label
 * @property {string} href - future/placeholder route
 * @property {React.ComponentType} icon - lucide-react icon
 * @property {boolean} [disabled] - renders inert with a "coming soon" tooltip
 */

/** @type {NavItem[]} */
export const NAV_ITEMS = [
  { value: "dashboard", label: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  { value: "reconstruction", label: "Reconstruction", href: "/reconstruction", icon: Waves },
  { value: "vertical-profile", label: "Vertical Profile", href: "/vertical-profile", icon: MoveVertical },
  { value: "hovmoller", label: "Hovmöller", href: "/hovmoller", icon: AreaChart },
  { value: "3d-ocean", label: "3D Ocean", href: "/3d-ocean", icon: Box },
  { value: "embedding-explorer", label: "Embedding Explorer", href: "/embedding-explorer", icon: Network },
  { value: "explainability", label: "Explainability", href: "/explainability", icon: Lightbulb },
  { value: "evaluation", label: "Evaluation", href: "/evaluation", icon: ClipboardCheck },
  { value: "data-quality", label: "Data Quality", href: "/data-quality", icon: ShieldCheck },
  { value: "system-status", label: "System Status", href: "/system-status", icon: Activity },
];

/**
 * Finds the NAV_ITEMS entry matching a pathname — exact match or a nested
 * route beneath it (e.g. "/reconstruction/123" still matches "Reconstruction").
 * @param {string | null | undefined} pathname
 * @returns {NavItem | undefined}
 */
export function getActiveNavItem(pathname) {
  if (!pathname) return undefined;
  return NAV_ITEMS.find((item) => pathname === item.href || pathname.startsWith(`${item.href}/`));
}