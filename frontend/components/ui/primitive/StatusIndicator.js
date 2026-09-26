// -----------------------------------------------------------------------------
// NEER Design System — UI component barrel
//
//   import { Button, Badge, StatusIndicator, Panel, Tabs, Slider, Select,
//            Modal, Drawer, Tooltip, MetricCard, ChartContainer,
//            LoadingSkeleton, ErrorState } from "@/components/ui";
//
// Directory structure (Phase 31E):
//   primitives/  Button, Badge, StatusIndicator      — smallest building blocks
//   layout/      Panel, Tabs                          — structural chrome
//   inputs/      Slider, Select                        — user input controls
//   overlays/    Modal, Drawer, Tooltip                — layered above content
//   data/        MetricCard, ChartContainer,
//                LoadingSkeleton, ErrorState            — data display/feedback
// -----------------------------------------------------------------------------

export { default as Button } from "./primitives/Button";
export { default as Badge } from "./primitives/Badge";
export { default as StatusIndicator } from "./primitives/StatusIndicator";

export { default as Panel } from "./layout/Panel";
export { default as Tabs } from "./layout/Tabs";

export { default as Slider } from "./inputs/Slider";
export { default as Select } from "./inputs/Select";

export { default as Modal } from "./overlays/Modal";
export { default as Drawer } from "./overlays/Drawer";
export { default as Tooltip } from "./overlays/Tooltip";

export { default as MetricCard } from "./data/MetricCard";
export { default as ChartContainer } from "./data/ChartContainer";
export { default as LoadingSkeleton } from "./data/LoadingSkeleton";
export { default as ErrorState } from "./data/ErrorState";