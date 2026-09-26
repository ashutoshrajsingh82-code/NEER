// -----------------------------------------------------------------------------
// NEER Design System — Panel
//
// The core "card" primitive — styled to resemble a scientific control-center
// module: hairline border, header/body/footer structure, optional icon and
// header actions. Purely presentational — no page-specific logic.
// -----------------------------------------------------------------------------

import { cn } from "@/lib/cn";

const EMPHASIS_CLASSES = {
  base: "neer-panel",
  raised: "neer-panel-raised",
  glass: "neer-glass shadow-panel",
  accent: "neer-panel neer-glow-ring",
};

/**
 * @param {string} title
 * @param {string} subtitle
 * @param {React.ComponentType} icon - optional lucide-react icon shown in the header
 * @param {React.ReactNode} headerActions - buttons/controls rendered top-right
 * @param {React.ReactNode} footer
 * @param {"base"|"raised"|"glass"|"accent"} emphasis - visual emphasis level
 */
export default function Panel({
  title,
  subtitle,
  icon: Icon,
  headerActions,
  footer,
  emphasis = "base",
  className,
  bodyClassName,
  children,
}) {
  const hasHeader = Boolean(title || subtitle || Icon || headerActions);

  return (
    <section
      className={cn(
        EMPHASIS_CLASSES[emphasis] ?? EMPHASIS_CLASSES.base,
        "flex flex-col overflow-hidden",
        className
      )}
    >
      {hasHeader && (
        <header className="neer-divider flex items-start justify-between gap-4 px-5 py-4">
          <div className="flex min-w-0 items-start gap-3">
            {Icon && (
              <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-border bg-surface-raised text-accent-400">
                <Icon size={18} strokeWidth={1.75} aria-hidden="true" />
              </span>
            )}
            <div className="min-w-0">
              {title && <h3 className="truncate text-h3 text-text-primary">{title}</h3>}
              {subtitle && <p className="mt-0.5 text-small text-text-muted">{subtitle}</p>}
            </div>
          </div>
          {headerActions && (
            <div className="flex shrink-0 items-center gap-2">{headerActions}</div>
          )}
        </header>
      )}

      <div className={cn("flex-1 px-5 py-4", bodyClassName)}>{children}</div>

      {footer && (
        <footer className="neer-divider bg-surface-sunken/40 px-5 py-3">{footer}</footer>
      )}
    </section>
  );
}