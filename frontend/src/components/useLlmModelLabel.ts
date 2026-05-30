"use client";

import { useEffect, useState } from "react";
import { health } from "@/lib/api";

// Cache the model label module-side so multiple consumers don't each
// hit /api/health on mount. First fetch wins; subsequent hook
// instances pick it up synchronously.
let cached: string | null = null;
let inflight: Promise<string | null> | null = null;

export function useLlmModelLabel(): string | null {
  const [llm, setLlm] = useState<string | null>(cached);

  useEffect(() => {
    if (cached !== null) return; // already captured by useState(cached) initializer
    if (!inflight) {
      inflight = health()
        .then((h) => {
          const provider = h.llm?.default_provider;
          const model = h.llm?.default_model;
          if (!model) return null;
          return provider ? `${provider}/${model}` : model;
        })
        .catch(() => null);
    }
    let cancelled = false;
    inflight.then((label) => {
      if (label && !cancelled) {
        cached = label;
        setLlm(label);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return llm;
}
