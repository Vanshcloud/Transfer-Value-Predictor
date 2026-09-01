/**
 * The one component whose correctness is a modelling claim rather than a
 * layout choice.
 *
 * SHAP values here are additive in *log* space, because the model fits
 * log1p(EUR). Rendering "age contributed −€2M" would be arithmetically false:
 * the same contribution is worth a different number of euros for a €500k
 * player and a €90M one. So the panel must show the multiplier, must never
 * show a euro figure per feature, and must keep saying so in words — a caveat
 * that quietly disappears in a refactor is worse than one that was never there.
 */
import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import Contributions from "./Contributions";
import type { Contribution } from "@/lib/api";

// Plotly is dynamically imported and never renders in jsdom. Stubbing it keeps
// these tests about the numbers, and the real SSR guard is covered by
// `next build` in CI, which is where that failure actually shows up.
vi.mock("./Chart", () => ({
  default: ({ title }: { title: string }) => <div data-testid="chart" aria-label={title} />,
}));

function contribution(over: Partial<Contribution> = {}): Contribution {
  return {
    feature: "numeric__goals",
    value: 12,
    shap_value: 0.3,
    effect_multiplier: Math.exp(0.3),
    direction: "increases",
    ...over,
  };
}

const positive = [
  contribution({ feature: "numeric__appearances", shap_value: 0.657, effect_multiplier: 1.93 }),
  contribution({ feature: "numeric__goals", shap_value: 0.3, effect_multiplier: 1.35 }),
];
const negative = [
  contribution({
    feature: "categorical__position_Attack",
    shap_value: -0.051,
    effect_multiplier: 0.95,
    direction: "decreases",
  }),
];

describe("Contributions", () => {
  it("shows the multiplier, which is exact at any player value", () => {
    render(<Contributions positive={positive} negative={negative} />);
    expect(screen.getByText(/×1\.93/)).toBeInTheDocument();
    expect(screen.getByText(/×0\.95/)).toBeInTheDocument();
  });

  it("never renders a euro figure per feature", () => {
    // The whole point. A per-feature euro amount would be a false statement.
    const { container } = render(<Contributions positive={positive} negative={negative} />);
    expect(container.textContent).not.toMatch(/€/);
  });

  it("converts the multiplier to a percentage with the right sign", () => {
    render(<Contributions positive={positive} negative={negative} />);
    expect(screen.getByText(/\+93%/)).toBeInTheDocument();
    expect(screen.getByText(/-5%/)).toBeInTheDocument();
  });

  it("keeps the caveat that these cannot be summed into euros", () => {
    render(<Contributions positive={positive} negative={negative} />);
    expect(screen.getByText(/cannot be summed into a euro figure/i)).toBeInTheDocument();
    expect(screen.getByText(/multiplicative, not additive/i)).toBeInTheDocument();
  });

  it("strips the ColumnTransformer prefix the model attaches", () => {
    render(<Contributions positive={positive} negative={negative} />);
    expect(screen.getByText("position Attack")).toBeInTheDocument();
    expect(screen.queryByText(/categorical__/)).not.toBeInTheDocument();
  });

  it("separates what raised the value from what lowered it", () => {
    render(<Contributions positive={positive} negative={negative} />);
    const raises = screen.getByRole("heading", { name: /raises the value/i }).parentElement!;
    const lowers = screen.getByRole("heading", { name: /lowers the value/i }).parentElement!;
    expect(within(raises).getByText("goals")).toBeInTheDocument();
    expect(within(lowers).getByText("position Attack")).toBeInTheDocument();
  });

  it("says so plainly when the family cannot be explained, rather than showing nothing", () => {
    render(<Contributions positive={[]} negative={[]} />);
    expect(screen.getByText(/cannot be explained with SHAP/i)).toBeInTheDocument();
    expect(screen.queryByTestId("chart")).not.toBeInTheDocument();
  });

  it("gives the chart an accessible name", () => {
    render(<Contributions positive={positive} negative={negative} />);
    expect(screen.getByTestId("chart")).toHaveAttribute(
      "aria-label",
      "Feature contributions to this prediction",
    );
  });
});
