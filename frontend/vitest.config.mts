/**
 * Vitest, not Jest: no separate transform pipeline to keep alive, and it reads
 * the JSX setting straight out of tsconfig.json.
 *
 * @vitejs/plugin-react is deliberately NOT used. Next 16 pulls in rolldown-vite
 * while vitest bundles its own vite, so the plugin's type resolves against one
 * copy and the config against the other — `tsc --noEmit` then fails on a
 * conflict that has nothing to do with this app. Vitest's esbuild transform
 * already handles `jsx: "react-jsx"`, and Fast Refresh is irrelevant to a
 * single non-watching CI run.
 *
 * jsdom rather than a real browser. These tests cover the logic that decides
 * *what* to render — the error mapping in lib/api, the state machine in
 * useAsync, the money formatting — and that each state renders with the right
 * accessible role. Whether Plotly draws the right pixels is not a question
 * jsdom can answer, and `next build` in CI already catches the SSR trap that is
 * the failure that actually breaks this app.
 */
import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  resolve: {
    alias: { "@": path.join(__dirname, "src") },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
