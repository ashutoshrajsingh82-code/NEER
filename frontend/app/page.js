import { redirect } from "next/navigation";

// -----------------------------------------------------------------------------
// NEER — root route
//
// The application shell has no "home" page of its own (Phase 32E scope is
// the shell, not any specific section) — it always opens into a section of
// NAV_ITEMS. Dashboard is the first item, so "/" simply forwards there.
// -----------------------------------------------------------------------------

export default function RootPage() {
  redirect("/dashboard");
}