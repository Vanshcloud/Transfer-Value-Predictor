import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// Unmount between tests so a query cannot match a component the previous test
// left behind — the failure mode where tests pass individually and not together.
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
