"use client";

// -----------------------------------------------------------------------------
// NEER Design System — Playground (development only)
//
// Visual QA surface for every component in the design system. This is NOT
// one of the real NEER application pages (dashboard/reconstruction/profile/
// map/metrics) — it exists purely so components can be eyeballed and
// interacted with in isolation while the design system is being built.
//
// Guarded to render nothing useful outside development, so it can't
// accidentally ship as a real route in production.
// -----------------------------------------------------------------------------

import { useState } from "react";
import {
  Activity,
  Droplet,
  Gauge,
  MapPin,
  Settings,
  Sparkles,
  Thermometer,
} from "lucide-react";
import {
  Badge,
  Button,
  ChartContainer,
  Drawer,
  ErrorState,
  LoadingSkeleton,
  MetricCard,
  Modal,
  Panel,
  Select,
  Slider,
  StatusIndicator,
  Tabs,
  Tooltip,
} from "@/components/ui";

const TAB_ITEMS = [
  { value: "overview", label: "Overview", icon: Sparkles },
  { value: "controls", label: "Controls", icon: Settings },
  { value: "disabled", label: "Disabled", disabled: true },
];

const DEPTH_OPTIONS = [
  { value: "surface", label: "Surface (0 m)" },
  { value: "thermocline", label: "Thermocline (~200 m)" },
  { value: "deep", label: "Deep ocean (>1000 m)" },
  { value: "trench", label: "Trench", disabled: true },
];

function Section({ title, description, children }) {
  return (
    <section className="flex flex-col gap-4">
      <div>
        <h2 className="text-h2 text-text-primary">{title}</h2>
        {description && <p className="mt-1 text-small text-text-muted">{description}</p>}
      </div>
      {children}
    </section>
  );
}

export default function DesignSystemPlayground() {
  if (process.env.NODE_ENV === "production") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-bg-base px-6 text-center">
        <p className="text-body text-text-muted">
          The design-system playground is only available in development.
        </p>
      </div>
    );
  }

  const [activeTab, setActiveTab] = useState("overview");
  const [depth, setDepth] = useState("thermocline");
  const [region, setRegion] = useState("");
  const [threshold, setThreshold] = useState(2.5);
  const [modalOpen, setModalOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [loadingDemo, setLoadingDemo] = useState(false);

  return (
    <div className="min-h-screen bg-bg-base bg-grid-subtle bg-grid px-6 py-10 text-text-primary sm:px-10">
      <div className="mx-auto flex max-w-6xl flex-col gap-12">
        <header>
          <Badge variant="accent">Development only</Badge>
          <h1 className="mt-3 text-display text-text-primary">NEER Design System Playground</h1>
          <p className="mt-2 max-w-2xl text-body text-text-muted">
            Every component from Phases 31A–31D, with their major variants, loading and error
            states, disabled states, and interactive behavior — for visual QA only.
          </p>
        </header>

        {/* Buttons & Badges ------------------------------------------------ */}
        <Section title="Button" description="Variants, sizes, icons, loading and disabled states.">
          <div className="flex flex-wrap items-center gap-3">
            <Button variant="primary">Primary</Button>
            <Button variant="secondary">Secondary</Button>
            <Button variant="ghost">Ghost</Button>
            <Button variant="danger">Danger</Button>
            <Button variant="primary" loading>
              Loading
            </Button>
            <Button variant="primary" disabled>
              Disabled
            </Button>
            <Button variant="secondary" icon={Settings}>
              With icon
            </Button>
            <Button variant="ghost" iconOnly icon={Settings} aria-label="Settings" />
            <Button variant="primary" size="sm">
              Small
            </Button>
            <Button variant="primary" size="lg">
              Large
            </Button>
          </div>
        </Section>

        <Section title="Badge" description="Semantic and scientific/data-status variants.">
          <div className="flex flex-wrap gap-2">
            <Badge>Default</Badge>
            <Badge variant="success">Success</Badge>
            <Badge variant="warning">Warning</Badge>
            <Badge variant="error">Error</Badge>
            <Badge variant="info">Info</Badge>
            <Badge variant="accent">Calibrated</Badge>
            <Badge variant="neutral">No signal</Badge>
          </div>
        </Section>

        <Section title="StatusIndicator" description="Live system/instrument states.">
          <div className="flex flex-wrap gap-5">
            <StatusIndicator status="online" />
            <StatusIndicator status="processing" />
            <StatusIndicator status="warning" />
            <StatusIndicator status="error" />
            <StatusIndicator status="offline" />
            <StatusIndicator status="unknown" />
          </div>
        </Section>

        {/* Layout ------------------------------------------------------------ */}
        <Section title="Panel" description="Emphasis levels and header/footer composition.">
          <div className="grid gap-4 sm:grid-cols-2">
            <Panel title="Base panel" subtitle="emphasis=&quot;base&quot;" icon={Gauge}>
              <p className="text-small text-text-secondary">Standard control-center card.</p>
            </Panel>
            <Panel
              title="Raised panel"
              subtitle="emphasis=&quot;raised&quot;"
              emphasis="raised"
              headerActions={<Button size="sm" variant="ghost">Action</Button>}
              footer={<p className="text-caption text-text-muted">Footer content</p>}
            >
              <p className="text-small text-text-secondary">With header action and footer.</p>
            </Panel>
          </div>
        </Section>

        <Section title="Tabs" description="Keyboard-navigable, animated active indicator, disabled tab.">
          <Tabs id="playground-tabs" items={TAB_ITEMS} value={activeTab} onChange={setActiveTab} />
          <div
            role="tabpanel"
            id={`playground-tabs-panel-${activeTab}`}
            aria-labelledby={`playground-tabs-tab-${activeTab}`}
            className="pt-4 text-small text-text-secondary"
          >
            Active tab: <span className="font-mono text-accent-300">{activeTab}</span>
          </div>
        </Section>

        {/* Inputs ------------------------------------------------------------ */}
        <Section title="Slider" description="Scientific numeric control with custom formatting.">
          <div className="grid gap-6 sm:grid-cols-2">
            <Slider
              label="Anomaly threshold"
              min={0}
              max={5}
              step={0.1}
              value={threshold}
              onChange={setThreshold}
              unit="°C"
            />
            <Slider label="Disabled example" min={0} max={100} value={40} disabled />
          </div>
        </Section>

        <Section title="Select" description="Keyboard-accessible dropdown with icon and error state.">
          <div className="grid gap-6 sm:grid-cols-2">
            <Select
              label="Depth layer"
              icon={MapPin}
              placeholder="Choose a depth…"
              options={DEPTH_OPTIONS}
              value={depth}
              onChange={setDepth}
            />
            <Select
              label="Region (required)"
              placeholder="Choose a region…"
              options={[{ value: "arabian-sea", label: "Arabian Sea" }]}
              value={region}
              onChange={setRegion}
              error={!region ? "A region must be selected." : undefined}
            />
          </div>
        </Section>

        {/* Overlays ------------------------------------------------------------ */}
        <Section title="Modal, Drawer & Tooltip" description="Backdrop, focus trap, escape-to-close, positioning.">
          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={() => setModalOpen(true)}>Open modal</Button>
            <Button variant="secondary" onClick={() => setDrawerOpen(true)}>
              Open drawer
            </Button>
            <Tooltip content="Rendered with hover, focus and a short delay">
              <Button variant="ghost">Hover or focus me</Button>
            </Tooltip>
          </div>

          <Modal
            open={modalOpen}
            onClose={() => setModalOpen(false)}
            title="Example modal"
            description="Demonstrates focus trapping and escape-to-close."
            footer={
              <>
                <Button variant="ghost" onClick={() => setModalOpen(false)}>
                  Cancel
                </Button>
                <Button onClick={() => setModalOpen(false)}>Confirm</Button>
              </>
            }
          >
            <p className="text-small text-text-secondary">Modal body content goes here.</p>
          </Modal>

          <Drawer open={drawerOpen} onClose={() => setDrawerOpen(false)} position="right" title="Example drawer">
            <p className="text-small text-text-secondary">Drawer body content goes here.</p>
          </Drawer>
        </Section>

        {/* Data & feedback ------------------------------------------------------------ */}
        <Section title="MetricCard" description="Generic metric display — not tied to any specific measurement.">
          <div className="grid gap-4 sm:grid-cols-3">
            <MetricCard
              title="Sea Surface Temp"
              value="28.4"
              unit="°C"
              icon={Thermometer}
              status="online"
              trend={{ direction: "up", value: "+0.3°C" }}
              description="vs. 7-day mean"
            />
            <MetricCard
              title="Model RMSE"
              value="0.142"
              icon={Activity}
              status="warning"
              trend={{ direction: "up", value: "+0.008", tone: "negative" }}
              description="Reconstruction error"
              secondaryValue={{ label: "MAE", value: "0.098" }}
            />
            <MetricCard
              title="Salinity"
              value="35.1"
              unit="PSU"
              icon={Droplet}
              status="offline"
              description="Sensor offline"
            />
          </div>
        </Section>

        <Section title="ChartContainer" description="Layout shell with loading, empty and fullscreen states.">
          <div className="grid gap-4 sm:grid-cols-2">
            <ChartContainer
              title="Depth profile"
              subtitle="Last 30 days"
              allowFullscreen
              legend={<span className="text-caption text-text-muted">Legend goes here</span>}
            >
              <div className="flex h-40 items-center justify-center rounded-md border border-dashed border-border text-caption text-text-muted">
                Chart content placeholder
              </div>
            </ChartContainer>
            <ChartContainer title="Loading example" loading />
          </div>
          <ChartContainer title="Empty example" empty emptyMessage="No readings for this date range." />
        </Section>

        <Section title="LoadingSkeleton" description="Six variants, all with a subtle pulse animation.">
          <div className="grid gap-4 sm:grid-cols-3">
            <LoadingSkeleton variant="metric" />
            <LoadingSkeleton variant="card" />
            <LoadingSkeleton variant="panel" />
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <LoadingSkeleton variant="list" count={3} />
            <LoadingSkeleton variant="text" lines={4} />
          </div>
        </Section>

        <Section title="ErrorState" description="Retry action and collapsible technical details.">
          <div className="grid gap-4 sm:grid-cols-2">
            <ErrorState onRetry={() => setLoadingDemo((v) => !v)} />
            <ErrorState
              title="Failed to fetch reconstruction output"
              message="The model service did not respond in time."
              details={"Error: timeout after 30000ms\n  at fetchReconstruction (client.js:42)"}
            />
          </div>
        </Section>
      </div>
    </div>
  );
}