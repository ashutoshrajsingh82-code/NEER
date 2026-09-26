"use client";

// -----------------------------------------------------------------------------
// NEER Application Shell — Sidebar
//
// Responsive behavior (Phase 32C):
//   < md   (mobile)  — hidden inline; rendered only inside the off-canvas Drawer
//   md–lg  (tablet)  — inline icon rail, collapsed by default but user-toggleable
//   >= lg  (desktop) — inline, expanded by default, user-toggleable
//
// Active state is derived from the current route (usePathname), not passed
// in by callers. Defaults to the shared NAV_ITEMS config.
// -----------------------------------------------------------------------------

import { useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { ChevronsLeft, ChevronsRight } from "lucide-react";
import { cn } from "@/lib/cn";
import { DURATION, EASE_OUT } from "@/lib/motion";
import { useMediaQuery } from "@/lib/useMediaQuery";
import { NAV_ITEMS } from "@/lib/navigation";
import Drawer from "@/components/ui/overlays/Drawer";
import Button from "@/components/ui/primitives/Button";
import SidebarItem from "./SidebarItem";
import { useShell } from "./ShellContext";

const RAIL_WIDTH = { expanded: 240, collapsed: 64 };

function isActive(pathname, href) {
  return pathname === href || (pathname?.startsWith(`${href}/`) ?? false);
}

function SidebarNav({ items, pathname, collapsed }) {
  return (
    <nav className="flex flex-1 flex-col gap-1 overflow-y-auto overflow-x-hidden px-2 py-3">
      {items.map((item) => (
        <SidebarItem key={item.value} item={item} active={isActive(pathname, item.href)} collapsed={collapsed} />
      ))}
    </nav>
  );
}

/**
 * @param {import("@/lib/navigation").NavItem[]} items - defaults to NAV_ITEMS
 * @param {React.ReactNode} footer - optional content pinned below the nav list (hidden while collapsed)
 */
export default function Sidebar({ items = NAV_ITEMS, footer, className }) {
  const pathname = usePathname();
  const { sidebarOpen, setSidebarOpen } = useShell();
  const isDesktop = useMediaQuery("(min-width: 1024px)");
  const [collapsed, setCollapsed] = useState(!isDesktop);
  const wasDesktop = useRef(isDesktop);

  // Auto-collapse when crossing the desktop breakpoint (tablet <-> desktop),
  // but don't fight a manual toggle made while staying within one range.
  useEffect(() => {
    if (wasDesktop.current !== isDesktop) {
      setCollapsed(!isDesktop);
      wasDesktop.current = isDesktop;
    }
  }, [isDesktop]);

  return (
    <>
      {/* Tablet + desktop: inline column, animated width on collapse/expand */}
      <motion.aside
        animate={{ width: collapsed ? RAIL_WIDTH.collapsed : RAIL_WIDTH.expanded }}
        transition={{ duration: DURATION.base, ease: EASE_OUT }}
        className={cn(
          "hidden shrink-0 flex-col overflow-hidden border-r border-border bg-surface-base md:flex",
          className
        )}
      >
        <SidebarNav items={items} pathname={pathname} collapsed={collapsed} />

        {footer && !collapsed && <div className="border-t border-border px-3 py-3">{footer}</div>}

        <div className={cn("flex border-t border-border p-2", collapsed ? "justify-center" : "justify-end")}>
          <Button
            variant="ghost"
            size="sm"
            iconOnly
            icon={collapsed ? ChevronsRight : ChevronsLeft}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-pressed={collapsed}
            onClick={() => setCollapsed((value) => !value)}
          />
        </div>
      </motion.aside>

      {/* Mobile: off-canvas drawer, always fully expanded */}
      <Drawer open={sidebarOpen} onClose={() => setSidebarOpen(false)} position="left" title="Navigation">
        <SidebarNav items={items} pathname={pathname} collapsed={false} />
        {footer && <div className="mt-3 border-t border-border pt-3">{footer}</div>}
      </Drawer>
    </>
  );
}