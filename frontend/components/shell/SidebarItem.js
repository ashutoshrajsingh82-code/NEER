"use client";

// -----------------------------------------------------------------------------
// NEER Application Shell — SidebarItem
//
// Renders one navigation entry using next/link (so client-side routing works
// correctly). Disabled items render as inert <span>s instead of links. In
// collapsed mode the item is wrapped in the Phase 31 Tooltip component so the
// label is still discoverable icon-only.
// -----------------------------------------------------------------------------

import Link from "next/link";
import { cn } from "@/lib/cn";
import Tooltip from "@/components/ui/overlays/Tooltip";

/**
 * @param {import("@/lib/navigation").NavItem} item
 * @param {boolean} active
 * @param {boolean} collapsed
 */
export default function SidebarItem({ item, active = false, collapsed = false }) {
  const Icon = item.icon;

  const sharedClasses = cn(
    "flex items-center gap-3 rounded-md px-3 py-2 text-small font-medium neer-transition",
    collapsed && "justify-center px-0"
  );

  const content = item.disabled ? (
    <span
      aria-disabled="true"
      tabIndex={-1}
      className={cn(sharedClasses, "cursor-not-allowed text-text-disabled opacity-40")}
    >
      {Icon && <Icon size={18} strokeWidth={1.75} aria-hidden="true" />}
      {!collapsed && <span className="truncate">{item.label}</span>}
    </span>
  ) : (
    <Link
      href={item.href}
      aria-current={active ? "page" : undefined}
      className={cn(
        sharedClasses,
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400",
        active
          ? "bg-surface-raised text-text-primary shadow-glow-sm"
          : "text-text-muted hover:bg-surface-raised/60 hover:text-text-secondary"
      )}
    >
      {Icon && <Icon size={18} strokeWidth={1.75} aria-hidden="true" />}
      {!collapsed && <span className="truncate">{item.label}</span>}
    </Link>
  );

  if (!collapsed) return content;

  return (
    <Tooltip content={item.disabled ? `${item.label} (coming soon)` : item.label} position="right">
      {content}
    </Tooltip>
  );
}