"use client";

// -----------------------------------------------------------------------------
// NEER Design System — Tabs
//
// Horizontal tab list. Controlled component — the caller owns `value` and
// renders whichever panel corresponds to it; Tabs only handles the tab strip
// itself, so it stays independent of any specific page's content.
// -----------------------------------------------------------------------------

import { useId, useRef } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/cn";
import { DURATION, EASE_OUT } from "@/lib/motion";

/**
 * @param {{value: string, label: string, icon?: React.ComponentType, disabled?: boolean}[]} items
 * @param {string} value - the currently active tab's value
 * @param {(value: string) => void} onChange
 */
export default function Tabs({ items, value, onChange, className }) {
  const groupId = useId();
  const tabRefs = useRef({});

  function focusAndSelect(targetIndex) {
    const enabled = items.filter((item) => !item.disabled);
    if (enabled.length === 0) return;
    const wrapped = enabled[(targetIndex + enabled.length) % enabled.length];
    tabRefs.current[wrapped.value]?.focus();
    onChange?.(wrapped.value);
  }

  function handleKeyDown(event, index) {
    if (event.key === "ArrowRight") {
      event.preventDefault();
      focusAndSelect(index + 1);
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      focusAndSelect(index - 1);
    } else if (event.key === "Home") {
      event.preventDefault();
      focusAndSelect(0);
    } else if (event.key === "End") {
      event.preventDefault();
      focusAndSelect(items.length - 1);
    }
  }

  return (
    <div
      role="tablist"
      aria-orientation="horizontal"
      className={cn("relative flex items-center gap-1 border-b border-border", className)}
    >
      {items.map((item, index) => {
        const isActive = item.value === value;
        const Icon = item.icon;
        return (
          <button
            key={item.value}
            ref={(el) => (tabRefs.current[item.value] = el)}
            role="tab"
            type="button"
            id={`${groupId}-tab-${item.value}`}
            aria-selected={isActive}
            aria-controls={`${groupId}-panel-${item.value}`}
            aria-disabled={item.disabled || undefined}
            disabled={item.disabled}
            tabIndex={isActive ? 0 : -1}
            onClick={() => !item.disabled && onChange?.(item.value)}
            onKeyDown={(event) => handleKeyDown(event, index)}
            className={cn(
              "relative flex items-center gap-1.5 rounded-t-sm px-4 py-2.5 text-small font-medium neer-transition",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400",
              isActive ? "text-text-primary" : "text-text-muted hover:text-text-secondary",
              item.disabled && "pointer-events-none cursor-not-allowed opacity-40"
            )}
          >
            {Icon && <Icon size={16} strokeWidth={1.75} aria-hidden="true" />}
            {item.label}
            {isActive && (
              <motion.span
                layoutId={`${groupId}-active-indicator`}
                className="absolute inset-x-2 -bottom-px h-0.5 rounded-full bg-accent-400 shadow-glow-sm"
                transition={{ duration: DURATION.base, ease: EASE_OUT }}
              />
            )}
          </button>
        );
      })}
    </div>
  );
}