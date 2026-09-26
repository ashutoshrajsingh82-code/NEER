"use client";

// -----------------------------------------------------------------------------
// NEER Design System — Tooltip
//
// Wraps a single child element and shows a small floating label on hover or
// keyboard focus. No page-specific logic — content/position are passed in.
// -----------------------------------------------------------------------------

import { cloneElement, useEffect, useId, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { cn } from "@/lib/cn";
import { scaleIn } from "@/lib/motion";

const POSITION_CLASSES = {
  top: "bottom-full left-1/2 mb-2 -translate-x-1/2",
  bottom: "top-full left-1/2 mt-2 -translate-x-1/2",
  left: "right-full top-1/2 mr-2 -translate-y-1/2",
  right: "left-full top-1/2 ml-2 -translate-y-1/2",
};

/**
 * @param {React.ReactNode} content - tooltip text/content; omit to disable
 * @param {React.ReactElement} children - a single focusable/hoverable element
 * @param {"top"|"bottom"|"left"|"right"} position
 * @param {number} delay - ms before the tooltip appears
 */
export default function Tooltip({ content, children, position = "top", delay = 300, className }) {
  const [visible, setVisible] = useState(false);
  const timeoutRef = useRef(null);
  const tooltipId = useId();

  useEffect(() => () => clearTimeout(timeoutRef.current), []);

  if (!content) return children;

  const show = () => {
    clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => setVisible(true), delay);
  };

  const hide = () => {
    clearTimeout(timeoutRef.current);
    setVisible(false);
  };

  const trigger = cloneElement(children, {
    onMouseEnter: (e) => {
      show();
      children.props.onMouseEnter?.(e);
    },
    onMouseLeave: (e) => {
      hide();
      children.props.onMouseLeave?.(e);
    },
    onFocus: (e) => {
      show();
      children.props.onFocus?.(e);
    },
    onBlur: (e) => {
      hide();
      children.props.onBlur?.(e);
    },
    "aria-describedby": tooltipId,
  });

  return (
    <span className="relative inline-flex">
      {trigger}
      <AnimatePresence>
        {visible && (
          <motion.span
            role="tooltip"
            id={tooltipId}
            initial="hidden"
            animate="visible"
            exit="exit"
            variants={scaleIn}
            className={cn(
              "pointer-events-none absolute z-toast whitespace-nowrap rounded-md border",
              "border-border-strong bg-surface-overlay px-2.5 py-1.5 text-caption text-text-primary shadow-raised",
              POSITION_CLASSES[position] ?? POSITION_CLASSES.top,
              className
            )}
          >
            {content}
          </motion.span>
        )}
      </AnimatePresence>
    </span>
  );
}