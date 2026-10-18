"use client";

/**
 * Only the newest request for a slot may change what is on screen.
 *
 * Responses do not arrive in the order requests were sent. Typing "Ha" then
 * "Haaland", or dragging a slider twice, sends two requests; if the first is
 * slower, it lands last and overwrites the answer to the question the user is
 * actually asking (audit F13). Aborting the fetch would not prevent that —
 * the response may already be in flight — and would not save the server any
 * work either, since the request has arrived.
 *
 * So each request takes a ticket for its slot, and checks it before touching
 * state. Taking a new ticket, or unmounting, makes every older ticket stale:
 *
 *     const isCurrent = begin("search");
 *     const data = await api.searchPlayers(term);
 *     if (!isCurrent()) return;
 *     setResults(data.results);
 *
 * Calling `begin(slot)` and discarding the result is how an action that
 * supersedes a request without sending one — clearing the search box — says so.
 */

import { useCallback, useEffect, useRef } from "react";

export function useLatest(): (slot?: string) => () => boolean {
  const tickets = useRef(new Map<string, number>());
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  return useCallback((slot = "default") => {
    const ticket = (tickets.current.get(slot) ?? 0) + 1;
    tickets.current.set(slot, ticket);
    return () => mounted.current && tickets.current.get(slot) === ticket;
  }, []);
}
