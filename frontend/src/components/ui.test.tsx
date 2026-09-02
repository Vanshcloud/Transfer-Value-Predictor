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
import { Avatar, Badge, Card, ClubTag, Empty, ErrorPanel, Loading, Stat } from "./ui";
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
    render(<ErrorPanel error={new ApiError("player_not_found", "no player with id 9", 404)} />);
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
    const { rerender } = render(<ErrorPanel error={new Error("down")} onRetry={onRetry} />);
    await userEvent.click(screen.getByRole("button", { name: /try again/i }));
    expect(onRetry).toHaveBeenCalledOnce();

    rerender(<ErrorPanel error={new Error("down")} />);
    expect(screen.queryByRole("button", { name: /try again/i })).not.toBeInTheDocument();
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
    expect(screen.getByRole("heading", { name: "Feature contributions" })).toBeInTheDocument();
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
    render(<Stat label="Temporal MAE" value="€4.44M" hint="held-out seasons" />);
    expect(screen.getByText("Temporal MAE")).toBeInTheDocument();
    expect(screen.getByText("€4.44M")).toBeInTheDocument();
    expect(screen.getByText("held-out seasons")).toBeInTheDocument();
  });

  it.each(["neutral", "positive", "negative", "warn"] as const)("renders the %s tone", (tone) => {
    render(<Badge tone={tone}>label</Badge>);
    expect(screen.getByText("label")).toBeInTheDocument();
  });
});

describe("Empty", () => {
  it("says why there is nothing, rather than rendering blank", () => {
    render(<Empty>No player matches “zzz”.</Empty>);
    expect(screen.getByText(/No player matches/)).toBeInTheDocument();
  });
});

describe("Avatar", () => {
  it.each([
    ["Erling Haaland", "EH"],
    ["Rodri", "R"],
    ["Vinicius Junior da Silva", "VJ"],
  ])("takes at most two initials from %s", (name, expected) => {
    const { container } = render(<Avatar name={name} seed={1} />);
    expect(container.textContent).toBe(expected);
  });

  it("stays hidden from screen readers, since the name sits beside it", () => {
    const { container } = render(<Avatar name="Erling Haaland" seed={1} />);
    expect(container.firstElementChild).toHaveAttribute("aria-hidden");
  });

  it("gives two players different hues", () => {
    const hue = (seed: number) =>
      render(<Avatar name="A B" seed={seed} />).container.firstElementChild?.getAttribute("style");
    expect(hue(1)).not.toBe(hue(2));
  });
});

describe("Avatar contrast", () => {
  /** Ratio of white initials against the disc, per WCAG relative luminance. */
  const contrastWithWhite = (style: string) => {
    const [r, g, b] = style.match(/\d+/g)!.map(Number).map((c) => c / 255);
    const lin = (c: number) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
    return 1.05 / (0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b) + 0.05);
  };

  it("keeps white initials readable on every hue a player id can produce", () => {
    // The golden angle walks all 360 hues, so a spot check would miss the
    // yellow-greens, which are exactly where white text gets hard to read.
    for (let seed = 1; seed <= 720; seed++) {
      const style = render(<Avatar name="A B" seed={seed} />)
        .container.firstElementChild!.getAttribute("style")!;
      expect(contrastWithWhite(style)).toBeGreaterThanOrEqual(4.5);
    }
  });
});

describe("ClubTag", () => {
  it("shows the club, with the flag standing in for the league", () => {
    const { container } = render(
      <ClubTag club="Arsenal FC" league="Premier League" country="England" />,
    );
    expect(screen.getByText("Arsenal FC")).toBeInTheDocument();
    expect(container.textContent).toContain("\u{1F3F4}");
  });

  it("reads the league out, since a flag alone announces nothing", () => {
    render(<ClubTag club="Arsenal FC" league="Premier League" country="England" />);
    // Two leagues in this dataset are both called Premier Liga, so a screen
    // reader that heard only the club would lose what the flag was carrying.
    expect(screen.getByText(", Premier League")).toHaveClass("sr-only");
  });

  it("falls back to the league when a player is between clubs", () => {
    render(<ClubTag club={null} league="Premier League" country="England" />);
    expect(screen.getByText("Premier League")).toBeInTheDocument();
  });

  it("renders nothing at all when neither is known", () => {
    const { container } = render(<ClubTag club={null} league={null} country={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("drops the flag rather than guessing at an unmapped country", () => {
    const { container } = render(
      <ClubTag club="Millonarios FC" league="Liga Betplay" country="Colombia?" />,
    );
    expect(container.textContent).toBe("Millonarios FC, Liga Betplay");
  });
});
