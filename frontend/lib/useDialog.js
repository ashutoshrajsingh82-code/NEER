"use client";

// -----------------------------------------------------------------------------
// NEER Design System — useDialog
//
// Shared behavior for overlay components (Modal, Drawer, and — as of Phase
// 32C — KPIDrawer's mobile bottom sheet): traps focus inside the dialog,
// closes on Escape, locks body scroll while open, and restores focus to the
// previously focused element on close. No page-specific logic — consumers
// pass `open`/`onClose` and get a ref to attach to the dialog panel.
// -----------------------------------------------------------------------------

import { useEffect, useRef } from "react";

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

// QA (Phase 32C): body-scroll locking is now reference-counted at module
// scope instead of each dialog saving/restoring `body.style.overflow`
// independently. With two overlays open at once (e.g. the mobile sidebar
// drawer AND the mobile KPI sheet), the old approach let whichever one
// closed first reset the scroll lock — even though the other overlay was
// still open. A shared counter only toggles the style at the 0↔1 transition.
let scrollLockCount = 0;
let previousBodyOverflow = "";

function lockBodyScroll() {
  if (scrollLockCount === 0) {
    previousBodyOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
  }
  scrollLockCount += 1;
}

function unlockBodyScroll() {
  scrollLockCount = Math.max(0, scrollLockCount - 1);
  if (scrollLockCount === 0) {
    document.body.style.overflow = previousBodyOverflow;
  }
}

export function useDialog({ open, onClose, initialFocusRef }) {
  const containerRef = useRef(null);
  const previouslyFocused = useRef(null);

  useEffect(() => {
    if (!open) return undefined;

    previouslyFocused.current = document.activeElement;
    const focusTarget = initialFocusRef?.current ?? containerRef.current;
    focusTarget?.focus();

    lockBodyScroll();

    function getFocusable() {
      if (!containerRef.current) return [];
      return Array.from(containerRef.current.querySelectorAll(FOCUSABLE_SELECTOR));
    }

    function handleKeyDown(event) {
      if (event.key === "Escape") {
        onClose?.();
        return;
      }
      if (event.key === "Tab") {
        const focusable = getFocusable();
        if (focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      unlockBodyScroll();
      previouslyFocused.current?.focus?.();
    };
  }, [open, onClose, initialFocusRef]);

  return containerRef;
}

export default useDialog;