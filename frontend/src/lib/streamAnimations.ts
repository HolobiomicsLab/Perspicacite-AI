"use client";

import { useEffect, useRef, useState } from "react";

// Even when the backend dumps a full answer in <100 ms (LLM provider
// returning everything in one chunk), we want the UI to *feel* like
// generation is happening. These two hooks decouple the *target*
// value (what the data says) from the *displayed* value (what the
// user sees), and interpolate the gap.

// Type a target string into the display at `charsPerFrame` per
// animation frame (≈60 fps → ~3 600 chars/s at default = 60). When
// the target grows further, we keep typing from where we left off.
// When the target shrinks (rare — a re-render after cancel), we snap.
export function useTypewriter(target: string, charsPerFrame = 60): string {
  const [shown, setShown] = useState(target);
  // Re-anchor when the target shrinks (new turn started).
  const anchorRef = useRef(target.length);

  useEffect(() => {
    if (target.length < anchorRef.current) {
      // New turn or reset — snap.
      anchorRef.current = target.length;
      setShown(target);
      return;
    }
    if (shown === target) return;
    let raf = 0;
    const tick = () => {
      setShown((prev) => {
        if (prev === target) return prev;
        const nextLen = Math.min(prev.length + charsPerFrame, target.length);
        const next = target.slice(0, nextLen);
        if (next.length < target.length) raf = requestAnimationFrame(tick);
        anchorRef.current = nextLen;
        return next;
      });
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, shown, charsPerFrame]);

  return shown;
}

// Progressively reveal a list over time: returns a slice of `items`
// that grows one element every `delayMs`. When `items` shrinks (new
// turn), we snap. This is the fix for "everything arrived as one
// block" — the data is in state immediately but only painted to the
// DOM progressively, which lets the user follow the cascade.
export function useStaggeredList<T>(items: T[], delayMs = 200): T[] {
  const [shownCount, setShownCount] = useState(items.length);
  const [prevLen, setPrevLen] = useState(items.length);
  const lastTickRef = useRef<number>(0);

  // Reset during render when the list shrank (turn restart) — avoids a
  // setState-in-effect cascade.
  if (items.length < prevLen) {
    setPrevLen(items.length);
    setShownCount(items.length);
  } else if (items.length !== prevLen) {
    setPrevLen(items.length);
  }

  useEffect(() => {
    if (shownCount >= items.length) return;
    // Caller passed delayMs ≤ 0 — they want everything now.
    if (delayMs <= 0) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- caller opted out of staggering; reveal all items at once
      setShownCount(items.length);
      return;
    }
    let raf = 0;
    const tick = (now: number) => {
      if (!lastTickRef.current) lastTickRef.current = now;
      const delta = now - lastTickRef.current;
      if (delta >= delayMs) {
        lastTickRef.current = now;
        setShownCount((c) => Math.min(c + 1, items.length));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      lastTickRef.current = 0;
    };
  }, [items.length, shownCount, delayMs]);

  return items.slice(0, shownCount);
}

// Lerp the displayed number towards `target` over ~durationMs. Cheap
// integer interpolation — good enough for token counters.
export function useAnimatedNumber(target: number, durationMs = 350): number {
  const [shown, setShown] = useState(target);
  const fromRef = useRef(target);
  const startRef = useRef<number>(0);

  useEffect(() => {
    if (shown === target) return;
    fromRef.current = shown;
    startRef.current = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - startRef.current) / durationMs);
      // Ease out — fast start, slow finish.
      const eased = 1 - Math.pow(1 - t, 2.2);
      const value = Math.round(
        fromRef.current + (target - fromRef.current) * eased,
      );
      setShown(value);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // We intentionally exclude `shown` from deps — anchoring is done
    // via fromRef so we don't restart the animation each render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, durationMs]);

  return shown;
}
