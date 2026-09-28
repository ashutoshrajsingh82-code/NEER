// -----------------------------------------------------------------------------
// NEER Frontend — Vitest config (Phase 33D)
//
// Scoped to `frontend/lib/**` for now: that's the framework-independent API
// client (`api.js`) this phase is about, plus its small React hook wrapper
// (`useApiRequest.js`). Page/component tests (jsdom, React Testing Library,
// ...) are a separate concern for whichever later phase actually builds
// those pages — not added here speculatively.
//
// Environment is plain "node": `api.js` only touches globals every modern
// Node already provides (fetch, URLSearchParams, AbortController) and has
// no DOM dependency, so there's no reason to pay jsdom's cost for these tests.
// -----------------------------------------------------------------------------
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  esbuild: {
    loader: "jsx",
    // Same automatic JSX runtime Next.js uses, so component files (which
    // don't `import React`) can be server-rendered in tests.
    jsx: "automatic",
    include: /.[jt]sx?$/,
    exclude: [],
  },
  test: {
    environment: "node",
    include: ["lib/**/*.test.js", "components/**/*.test.js"],
    restoreMocks: true,
  },
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./", import.meta.url)),
    },
  },
});