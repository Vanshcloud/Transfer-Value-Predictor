/**
 * The shared states, and the accessible roles they carry.
 *
 * These exist so that "the backend is down" is distinguishable from "the panel
 * is empty" is distinguishable from "still loading" — for a screen reader as
 * well as a sighted user. That distinction is a rendering decision, which is
 * exactly what jsdom can check.
 */
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Badge, Card, Empty, ErrorPanel, Loading, Stat } from "./ui";
import { ApiError } from "@/lib/api";

describe("Loading", () => {
  it("announces itself politely instead of only animating", () => {
    render(<Loading label="Searching" />);
    const status = screen.getByRole("status");
    expect(status).toHaveAttribute("aria-live", "polite");
    // A pulsing rectangle says nothing to a screen reader.
    expect(screen.getByText("Searching")).toBeInTheDocument();
  });
});

describe("ErrorPanel", () => {
  it("uses role=alert so the failure is announced, not just shown", () => {
    render(<ErrorPanel error={new Error("it broke")} />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("it broke")).toBeInTheDocument();
  });

  it("shows the server's code so a bug report can name it", () => {
    render(
      <ErrorPanel
        error={new ApiError("player_not_found", "no player with id 9", 404)}
      />,
    );
    expect(screen.getByText("player_not_found")).toBeInTheDocument();
    expect(screen.getByText("no player with id 9")).toBeInTheDocument();
  });

  it("labels a non-Error as unexpected rather than crashing on it", () => {
    render(<ErrorPanel error="a bare string" />);
    expect(screen.getByText("unexpected_error")).toBeInTheDocument();
    expect(screen.getByText("a bare string")).toBeInTheDocument();
  });

  it("offers retry only when there is something to retry", async () => {
    const onRetry = vi.fn();
    const { rerender } = render(
      <ErrorPanel error={new Error("down")} onRetry={onRetry} />,
    );
    await userEvent.click(screen.getByRole("button", { name: /try again/i }));
    expect(onRetry).toHaveBeenCalledOnce();

    rerender(<ErrorPanel error={new Error("down")} />);
    expect(
      screen.queryByRole("button", { name: /try again/i }),
    ).not.toBeInTheDocument();
  });

  it("is reachable by keyboard", async () => {
    const onRetry = vi.fn();
    render(<ErrorPanel error={new Error("down")} onRetry={onRetry} />);
    await userEvent.tab();
    expect(screen.getByRole("button", { name: /try again/i })).toHaveFocus();
    await userEvent.keyboard("{Enter}");
    expect(onRetry).toHaveBeenCalledOnce();
  });
});

describe("Card", () => {
  it("renders its title as a heading, so the page has an outline", () => {
    render(
      <Card title="Feature contributions" subtitle="What moved this prediction">
        <p>body</p>
      </Card>,
    );
    expect(
      screen.getByRole("heading", { name: "Feature contributions" }),
    ).toBeInTheDocument();
    expect(screen.getByText("What moved this prediction")).toBeInTheDocument();
  });

  it("omits the header entirely when there is no title", () => {
    render(
      <Card>
        <p>body</p>
      </Card>,
    );
    expect(screen.queryByRole("heading")).not.toBeInTheDocument();
  });
});

describe("Stat and Badge", () => {
  it("keeps the label with its value", () => {
    render(
      <Stat label="Temporal MAE" value="€4.44M" hint="held-out seasons" />,
    );
    expect(screen.getByText("Temporal MAE")).toBeInTheDocument();
    expect(screen.getByText("€4.44M")).toBeInTheDocument();
    expect(screen.getByText("held-out seasons")).toBeInTheDocument();
  });

  it.each(["neutral", "positive", "negative", "warn"] as const)(
    "renders the %s tone",
    (tone) => {
      render(<Badge tone={tone}>label</Badge>);
      expect(screen.getByText("label")).toBeInTheDocument();
    },
  );
});

describe("Empty", () => {
  it("says why there is nothing, rather than rendering blank", () => {
    render(<Empty>No player matches “zzz”.</Empty>);
    expect(screen.getByText(/No player matches/)).toBeInTheDocument();
  });
});
