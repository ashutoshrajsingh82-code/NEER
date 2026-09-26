"use client";

// -----------------------------------------------------------------------------
// NEER Design System — Select
//
// A custom listbox-based dropdown (rather than a bare native <select>) so
// styling stays consistent with the rest of the dark control-center theme.
// Fully keyboard operable and controlled by the caller via value/onChange.
// -----------------------------------------------------------------------------

import { useEffect, useId, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "@/lib/cn";
import { scaleIn } from "@/lib/motion";

/**
 * @param {string} label
 * @param {string} placeholder
 * @param {{value: string, label: string, disabled?: boolean}[]} options
 * @param {string} value
 * @param {(value: string) => void} onChange
 * @param {boolean} disabled
 * @param {string} error - error message; also switches the trigger to an error style
 * @param {React.ComponentType} icon - optional leading icon in the trigger
 */
export default function Select({
  label,
  placeholder = "Select…",
  options = [],
  value,
  onChange,
  disabled = false,
  error,
  icon: Icon,
  className,
}) {
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const containerRef = useRef(null);
  const listRef = useRef(null);
  const listboxId = useId();
  const selected = options.find((option) => option.value === value);

  useEffect(() => {
    function handleClickOutside(event) {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    if (open) listRef.current?.focus();
  }, [open]);

  function openList() {
    if (disabled) return;
    const currentIndex = options.findIndex((option) => option.value === value);
    setActiveIndex(currentIndex >= 0 ? currentIndex : 0);
    setOpen(true);
  }

  function handleTriggerKeyDown(event) {
    if (["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
      event.preventDefault();
      openList();
    }
  }

  function selectOption(option) {
    if (!option || option.disabled) return;
    onChange?.(option.value);
    setOpen(false);
  }

  function handleListKeyDown(event) {
    if (event.key === "Escape") {
      event.preventDefault();
      setOpen(false);
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((index) => Math.min(index + 1, options.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex((index) => Math.max(index - 1, 0));
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectOption(options[activeIndex]);
    }
  }

  return (
    <div ref={containerRef} className={cn("relative w-full", className)}>
      {label && <label className="mb-1.5 block text-small font-medium text-text-secondary">{label}</label>}

      <button
        type="button"
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listboxId}
        onClick={() => (open ? setOpen(false) : openList())}
        onKeyDown={handleTriggerKeyDown}
        className={cn(
          "flex w-full items-center justify-between gap-2 rounded-md border bg-surface-base px-3.5 py-2.5 text-left text-body neer-transition",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400",
          error ? "border-error-border" : "border-border-strong hover:border-border-accent",
          disabled && "cursor-not-allowed opacity-45"
        )}
      >
        <span className="flex min-w-0 items-center gap-2">
          {Icon && <Icon size={16} strokeWidth={1.75} className="shrink-0 text-text-muted" aria-hidden="true" />}
          <span className={cn("truncate", selected ? "text-text-primary" : "text-text-muted")}>
            {selected ? selected.label : placeholder}
          </span>
        </span>
        <ChevronDown
          size={16}
          strokeWidth={1.75}
          className={cn("shrink-0 text-text-muted neer-transition", open && "rotate-180")}
          aria-hidden="true"
        />
      </button>

      <AnimatePresence>
        {open && (
          <motion.ul
            ref={listRef}
            role="listbox"
            id={listboxId}
            tabIndex={-1}
            aria-activedescendant={activeIndex >= 0 ? `${listboxId}-option-${activeIndex}` : undefined}
            initial="hidden"
            animate="visible"
            exit="exit"
            variants={scaleIn}
            onKeyDown={handleListKeyDown}
            className="absolute z-overlay mt-1.5 max-h-60 w-full overflow-auto rounded-md border border-border-strong bg-surface-overlay py-1 shadow-raised focus:outline-none"
          >
            {options.map((option, index) => {
              const isSelected = option.value === value;
              return (
                <li
                  key={option.value}
                  id={`${listboxId}-option-${index}`}
                  role="option"
                  aria-selected={isSelected}
                  onMouseEnter={() => setActiveIndex(index)}
                  onClick={() => selectOption(option)}
                  className={cn(
                    "flex cursor-pointer items-center justify-between gap-2 px-3.5 py-2 text-small",
                    index === activeIndex ? "bg-surface-raised text-text-primary" : "text-text-secondary",
                    option.disabled && "cursor-not-allowed opacity-40"
                  )}
                >
                  <span className="truncate">{option.label}</span>
                  {isSelected && <Check size={14} strokeWidth={2} className="shrink-0 text-accent-400" aria-hidden="true" />}
                </li>
              );
            })}
          </motion.ul>
        )}
      </AnimatePresence>

      {error && <p className="mt-1.5 text-caption text-error">{error}</p>}
    </div>
  );
}