/**
 * The compare page's selection, encoded in the query string.
 *
 * Both sides used to live in React state alone, so opening a player's page and
 * pressing Back returned an empty form — the component had unmounted and
 * remounted with no memory of who was being compared. The pair is the page's
 * entire meaning, so it belongs in the address bar, which also makes a
 * comparison a link somebody can send.
 *
 * Pure functions, kept out of the component so the parsing has somewhere to be
 * tested. The effects that call `window.history` stay in the page.
 */

/** Player ids named in a query string, ignoring anything that is not one. */
export function readPair(search: string): { a: number | null; b: number | null } {
  const params = new URLSearchParams(search);
  const one = (key: string) => {
    const raw = params.get(key);
    if (raw === null) return null;
    const id = Number(raw);
    // A hand-edited or stale link should leave an empty picker rather than
    // request player NaN: there is nothing for the reader to fix.
    return Number.isInteger(id) && id > 0 ? id : null;
  };
  return { a: one("a"), b: one("b") };
}

/**
 * The query string for a pair, given a path. Either side may be absent —
 * clearing one player must drop its parameter rather than leave a stale id
 * that the next reload would resurrect.
 */
export function writePair(
  pathname: string,
  a: number | null | undefined,
  b: number | null | undefined,
): string {
  const params = new URLSearchParams();
  if (a != null) params.set("a", String(a));
  if (b != null) params.set("b", String(b));
  const query = params.toString();
  return query ? `${pathname}?${query}` : pathname;
}
