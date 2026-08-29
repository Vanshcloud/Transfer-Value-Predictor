"use client";

/**
 * Light/dark toggle.
 *
 * The current theme lives on <html> as a class, applied before paint by the
 * inline script in the layout — an effect would render light first and flash
 * white on every dark-mode reload. Reading it back is subscribing to state
 * outside React, so `useSyncExternalStore` is the right tool: it also gives a
 * server snapshot, which keeps hydration honest.
 */

import { useCallback, useSyncExternalStore } from "react";

function subscribe(onChange: () => void): () => void {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class"],
  });
  return () => observer.disconnect();
}

function isDark(): boolean {
  return document.documentElement.classList.contains("dark");
}

/** The server cannot know the viewer's theme; light is the safe assumption. */
function serverSnapshot(): boolean {
  return false;
}

export function useIsDark(): boolean {
  return useSyncExternalStore(subscribe, isDark, serverSnapshot);
}

export default function ThemeToggle() {
  const dark = useIsDark();

  const toggle = useCallback(() => {
    const next = !document.documentElement.classList.contains("dark");
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("theme", next ? "dark" : "light");
    } catch {
      // Private browsing can refuse storage. The toggle still works for this
      // page; it just will not be remembered.
    }
  }, []);

  return (
    <button
      onClick={toggle}
      aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
      className="rounded-lg border border-slate-200 px-2.5 py-1.5 text-sm hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
    >
      {dark ? "\u2600" : "\u263e"}
    </button>
  );
}
