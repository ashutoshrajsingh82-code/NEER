"use client";

// -----------------------------------------------------------------------------
// NEER Application Shell — useMediaQuery
//
// Tracks a CSS media query's match state. Used where a component needs to
// change *behavior* (not just styling — that stays in Tailwind classes) based
// on viewport size, e.g. Sidebar's default collapsed state or KPIDrawer
// switching between an inline panel and a mobile bottom sheet.
//
// Starts `false` on both server and first client render (so hydration never
// mismatches), then syncs to the real value after mount.
// -----------------------------------------------------------------------------

import { useEffect, useState } from "react";

export function useMediaQuery(query) {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    const mediaQueryList = window.matchMedia(query);
    setMatches(mediaQueryList.matches);

    function handleChange(event) {
      setMatches(event.matches);
    }

    mediaQueryList.addEventListener("change", handleChange);
    return () => mediaQueryList.removeEventListener("change", handleChange);
  }, [query]);

  return matches;
}

export default useMediaQuery;