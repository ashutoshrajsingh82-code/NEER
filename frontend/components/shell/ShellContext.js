"use client";

// -----------------------------------------------------------------------------
// NEER Application Shell — ShellContext
//
// Cross-cutting state that TopNavigation, Sidebar and InspectionPanel all need
// to agree on: whether the off-canvas sidebar/inspection drawers are open on
// small screens. Kept minimal — desktop collapse state and KPIDrawer's
// expand/collapse are local to those components, since nothing else needs
// to react to them.
// -----------------------------------------------------------------------------

import { createContext, useContext, useState } from "react";

const ShellContext = createContext(null);

export function ShellProvider({ children }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [inspectionOpen, setInspectionOpen] = useState(false);

  const value = { sidebarOpen, setSidebarOpen, inspectionOpen, setInspectionOpen };

  return <ShellContext.Provider value={value}>{children}</ShellContext.Provider>;
}

export function useShell() {
  const context = useContext(ShellContext);
  if (!context) {
    throw new Error("useShell must be used within an <AppShell>.");
  }
  return context;
}