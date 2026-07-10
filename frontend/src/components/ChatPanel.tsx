"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { DatabasePicker } from "./DatabasePicker";
import { SourcePill } from "./SourcePill";
import { StatusBar, type ChatPhase } from "./StatusBar";
import { ThinkingSteps } from "./ThinkingSteps";
import { Markdown } from "./Markdown";
import {
  streamChat,
  cancelChat,
  estimateTokens,
  type ChatSource,
  type ThinkingStep,
} from "@/lib/chat";
import { useTypewriter, useStaggeredList } from "@/lib/streamAnimations";
import { estimateCostUsd, formatUsd } from "@/lib/cost";
import { useLlmModelLabel } from "./useLlmModelLabel";
import type { RAGMode } from "@/lib/modes";
import { MODES, accentClasses } from "@/lib/modes";
import { DATABASES, DEFAULT_DATABASES, type DatabaseId } from "@/lib/databases";
import { DatabaseGlyph } from "./DatabaseGlyph";
import { conversations as convApi, kb as kbApi, type ConvMessage, type KBSummary } from "@/lib/api";
import { loadPreferences } from "@/lib/preferences";

type Turn = {
  id: string;
  role: "user" | "assistant";
  mode?: RAGMode;
  text: string;
  sources?: ChatSource[];
  steps?: ThinkingStep[];
  streaming?: boolean;
  startedAt?: number;
  elapsedMs?: number;
  papersFound?: number;
  tokensIn?: number;
  tokensOut?: number;
  phase?: ChatPhase;
  error?: string;
  iterationCount?: number;
  completionReason?: string;
  diagnostic?: Record<string, unknown> | null;
};

function nid(): string {
  return Math.random().toString(36).slice(2, 10);
}

const EXAMPLE_PROMPTS = [
  "What are critique tokens in Self-RAG?",
  "Compare retrieval-augmented generation with corrective RAG.",
  "Summarise the state-of-the-art on multi-modal scientific search.",
  "Which mass-spectrometry foundation models exist in 2026?",
];

// Heuristic: pick a status phase from the latest thinking step.
function phaseFromStep(step?: ThinkingStep): ChatPhase | undefined {
  if (!step) return undefined;
  if (step.kind === "provider_progress") return "retrieving";
  if (step.kind === "batch_progress") return "retrieving";
  if (step.kind === "query_rephrased") return "sending";
  if (step.kind === "status") {
    const m = step.label.toLowerCase();
    if (m.includes("generat") || m.includes("synth")) return "synthesizing";
    if (m.includes("retriev") || m.includes("search") || m.includes("query"))
      return "retrieving";
  }
  return undefined;
}

export function ChatPanel({
  initialConversationId,
}: {
  initialConversationId?: string;
}) {
  const [mode, setMode] = useState<RAGMode>("basic");
  const [draft, setDraft] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [databases, setDatabases] = useState<DatabaseId[]>(DEFAULT_DATABASES);
  const [showDbPicker, setShowDbPicker] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [maxPapers, setMaxPapers] = useState<number>(10);
  // Hybrid retrieval weight: 0 = pure BM25, 0.5 = default, 1 = pure vector
  const [hybridWeight, setHybridWeight] = useState<number>(0.5);
  const [kbName, setKbName] = useState<string | null>(null);
  const [kbList, setKbList] = useState<KBSummary[]>([]);

  // Honour user preferences on first mount (default mode, max papers,
  // default DBs, default KB). Composer toggles still override per-request.
  useEffect(() => {
    const p = loadPreferences();
    setMode(p.defaultMode);
    setDatabases(p.defaultDatabases);
    setMaxPapers(p.maxPapers);
    setKbName(p.defaultKbName);
  }, []);

  // Load the user's knowledge bases for the composer KB picker.
  useEffect(() => {
    kbApi.list().then(setKbList).catch(() => setKbList([]));
  }, []);
  const conversationIdRef = useRef<string | undefined>(initialConversationId);
  const abortRef = useRef<AbortController | null>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);

  const currentMode = useMemo(() => MODES.find((m) => m.id === mode)!, [mode]);
  const accent = accentClasses(currentMode.accent);

  // Resume an existing conversation when /chat/[id] is opened.
  useEffect(() => {
    conversationIdRef.current = initialConversationId;
    if (!initialConversationId) {
      setTurns([]);
      return;
    }
    let cancelled = false;
    setLoadingHistory(true);
    convApi
      .get(initialConversationId)
      .then((conv) => {
        if (cancelled) return;
        // Restored messages have no mode/steps stored backend-side.
        // Tag assistant turns with the current default so the agentic
        // trail header still renders (it falls back to the planned
        // phases for that mode, even though no real events arrived).
        const restored: Turn[] = (conv.messages ?? []).map((m: ConvMessage) => ({
          id: nid(),
          role: m.role === "user" ? "user" : "assistant",
          mode: m.role === "assistant" ? mode : undefined,
          text: m.content,
        }));
        setTurns(restored);
      })
      .catch(() => {
        if (!cancelled) setTurns([]);
      })
      .finally(() => {
        if (!cancelled) setLoadingHistory(false);
      });
    return () => {
      cancelled = true;
    };
  }, [initialConversationId]);

  // Auto-scroll only when a NEW turn is added — not on every token —
  // and use window scroll because the chat panel's inner div is
  // taller than the viewport (the document body is what actually
  // scrolls). Smoothly bring the latest turn into view.
  const lastTurnIdRef = useRef<string | undefined>(undefined);
  useEffect(() => {
    const el = scrollerRef.current;
    const newest = turns[turns.length - 1];
    if (!el || !newest || newest.id === lastTurnIdRef.current) return;
    lastTurnIdRef.current = newest.id;
    // Find the newest <li> rendered for this turn and scroll it into view.
    const lis = el.querySelectorAll(":scope > ol > li");
    const lastLi = lis[lis.length - 1] as HTMLElement | undefined;
    if (lastLi) {
      lastLi.scrollIntoView({ behavior: "smooth", block: "start" });
    } else {
      // Fallback for the page-level scroll.
      window.scrollTo({
        top: document.documentElement.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [turns]);

  // Tick the elapsed counter on the streaming turn.
  useEffect(() => {
    if (!streaming) return;
    const interval = setInterval(() => {
      setTurns((ts) =>
        ts.map((t) =>
          t.streaming && t.startedAt
            ? { ...t, elapsedMs: Date.now() - t.startedAt }
            : t,
        ),
      );
    }, 250);
    return () => clearInterval(interval);
  }, [streaming]);

  // Esc to cancel during streaming.
  useEffect(() => {
    if (!streaming) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        cancel();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [streaming]);

  const submit = useCallback(
    async (queryArg?: string) => {
      const q = (queryArg ?? draft).trim();
      if (!q || streaming) return;

      const tokensIn = estimateTokens(q);
      const userTurn: Turn = { id: nid(), role: "user", text: q, mode };
      const asstTurn: Turn = {
        id: nid(),
        role: "assistant",
        mode,
        text: "",
        streaming: true,
        startedAt: Date.now(),
        elapsedMs: 0,
        tokensIn,
        tokensOut: 0,
        phase: "sending",
        steps: [],
      };
      setTurns((t) => [...t, userTurn, asstTurn]);
      setDraft("");
      setStreaming(true);
      // Auto-collapse the database panel on submit so the answer + sources
      // get the full window. The user can reopen it from the favicon row.
      setShowDbPicker(false);

      const ctrl = new AbortController();
      abortRef.current = ctrl;

      try {
        for await (const ev of streamChat({
          query: q,
          mode,
          conversationId: conversationIdRef.current,
          databases,
          kbName: kbName ?? undefined,
          maxPapers,
          // Only send when mode uses hybrid retrieval AND weight is non-default
          ...(["advanced", "deep_research"].includes(mode) && hybridWeight !== 0.5
            ? { bm25Weight: parseFloat((1 - hybridWeight).toFixed(2)), vectorWeight: parseFloat(hybridWeight.toFixed(2)) }
            : {}),
          signal: ctrl.signal,
        })) {
          if (ev.kind === "token") {
            setTurns((ts) =>
              ts.map((t) => {
                if (t.id !== asstTurn.id) return t;
                const text = t.text + ev.text;
                return {
                  ...t,
                  text,
                  tokensOut: estimateTokens(text),
                  phase: "synthesizing",
                };
              }),
            );
          } else if (ev.kind === "meta") {
            // Sources arrive one-per-event in basic mode (each {"type":"source"}
            // frame) and as a bulk list in agentic/papers_found frames. Both
            // hit this branch as `kind: "meta"`. We append+dedup so the
            // single-source case doesn't overwrite earlier ones (which used
            // to collapse 5 retrieved papers down to "Sources · 1").
            setTurns((ts) =>
              ts.map((t) => {
                if (t.id !== asstTurn.id) return t;
                let nextSources = t.sources;
                if (ev.sources && ev.sources.length > 0) {
                  // Use `||` rather than `??` so empty-string fields fall
                  // through. The backend can emit `doi: ""` for papers
                  // without a registered DOI; `??` would treat "" as a real
                  // key and collapse every empty-DOI paper into one pill.
                  const keyOf = (s: ChatSource) =>
                    s.doi || s.paper_id || s.url || s.title || "";
                  const existing = t.sources ?? [];
                  const seen = new Set(
                    existing.map(keyOf).filter((k) => k),
                  );
                  const merged = [...existing];
                  for (const s of ev.sources) {
                    const key = keyOf(s);
                    // Empty key → no identifying field at all; keep the
                    // paper rather than silently collapsing it.
                    if (!key || !seen.has(key)) {
                      if (key) seen.add(key);
                      merged.push(s);
                    }
                  }
                  nextSources = merged;
                }
                return {
                  ...t,
                  sources: nextSources,
                  papersFound: ev.papers_found ?? t.papersFound,
                };
              }),
            );
          } else if (ev.kind === "thinking") {
            setTurns((ts) =>
              ts.map((t) =>
                t.id === asstTurn.id
                  ? {
                      ...t,
                      steps: [...(t.steps ?? []), ev.step],
                      phase: phaseFromStep(ev.step) ?? t.phase,
                    }
                  : t,
              ),
            );
          } else if (ev.kind === "metadata") {
            setTurns((ts) =>
              ts.map((t) =>
                t.id === asstTurn.id
                  ? {
                      ...t,
                      iterationCount: ev.iteration_count,
                      completionReason: ev.completion_reason,
                      diagnostic: ev.diagnostic,
                    }
                  : t,
              ),
            );
          } else if (ev.kind === "done") {
            if (ev.conversation_id) {
              const newId = ev.conversation_id;
              conversationIdRef.current = newId;
              // Push the URL into the bar without going through Next's
              // router — `router.replace` re-runs the load-from-backend
              // effect and would clobber the in-flight turn (mode,
              // sources, steps all get lost when the conversation API
              // returns just text). history.replaceState skips that.
              if (!initialConversationId && typeof window !== "undefined") {
                window.history.replaceState(
                  null,
                  "",
                  `/chat/${encodeURIComponent(newId)}`,
                );
              }
            }
            setTurns((ts) =>
              ts.map((t) =>
                t.id === asstTurn.id
                  ? {
                      ...t,
                      streaming: false,
                      phase: "done",
                      text: ev.answer && !t.text ? ev.answer : t.text,
                      tokensOut: estimateTokens(
                        ev.answer && !t.text ? ev.answer : t.text,
                      ),
                    }
                  : t,
              ),
            );
          } else if (ev.kind === "error") {
            setTurns((ts) =>
              ts.map((t) =>
                t.id === asstTurn.id
                  ? {
                      ...t,
                      streaming: false,
                      phase: "error",
                      error: ev.message,
                    }
                  : t,
              ),
            );
          }
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : "stream interrupted";
        const wasAborted =
          err instanceof DOMException && err.name === "AbortError";
        setTurns((ts) =>
          ts.map((t) =>
            t.id === asstTurn.id
              ? {
                  ...t,
                  streaming: false,
                  phase: wasAborted ? "cancelled" : "error",
                  error: wasAborted ? undefined : msg,
                }
              : t,
          ),
        );
      } finally {
        setStreaming(false);
        abortRef.current = null;
        setTurns((ts) =>
          ts.map((t) =>
            t.id === asstTurn.id ? { ...t, streaming: false } : t,
          ),
        );
      }
    },
    [draft, mode, streaming, databases, initialConversationId, kbName, maxPapers, hybridWeight],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    if (conversationIdRef.current) cancelChat(conversationIdRef.current);
  }, []);

  const hasMessages = turns.length > 0;
  const lastAssistant = [...turns].reverse().find((t) => t.role === "assistant");
  const livePhase: ChatPhase = streaming
    ? lastAssistant?.phase ?? "sending"
    : lastAssistant?.phase ?? "idle";

  // Cumulative tokens — sum across all COMPLETED assistant turns
  // (skip the one currently streaming, since its counters are shown
  // separately as the "current" pair).
  const historyTokens = useMemo(() => {
    let i = 0;
    let o = 0;
    for (const t of turns) {
      if (t.role !== "assistant") continue;
      if (t.streaming) continue;
      i += t.tokensIn ?? 0;
      o += t.tokensOut ?? 0;
    }
    return { in: i, out: o };
  }, [turns]);

  return (
    <section className="relative mx-auto flex w-full max-w-5xl flex-1 flex-col px-4 md:px-6">
      {/* Transcript or hero empty-state. Bottom padding leaves room
          so the last turn's footer doesn't hide behind the composer
          or sit on top of the status pill above it. */}
      <div ref={scrollerRef} className="flex-1 overflow-y-auto px-1 pb-[210px] pt-6">
        {loadingHistory ? (
          <p className="py-12 text-center text-sm text-[var(--text-muted)]">
            Loading conversation…
          </p>
        ) : !hasMessages ? (
          <HeroEmptyState mode={mode} onPick={(prompt) => setDraft(prompt)} />
        ) : (
          <ol className="flex flex-col gap-6">
            {turns.map((t) =>
              t.role === "user" ? (
                <UserMessage key={t.id} turn={t} />
              ) : (
                <AssistantMessage key={t.id} turn={t} streaming={streaming} />
              ),
            )}
          </ol>
        )}
      </div>

      {/* Compose — `fixed` to the viewport bottom so absolutely nothing
          above can shift its position during streaming. The wrapper
          mirrors the section's max-width and horizontal padding so
          it stays aligned with the messages column. */}
      <div className="fixed bottom-0 left-0 right-0 z-20 border-t border-[var(--border)] bg-[var(--bg)]/95 backdrop-blur md:left-[272px]">
        <div className="relative mx-auto w-full max-w-5xl px-4 pb-4 pt-3 md:px-6">
        {/* Status pill — sits directly ABOVE the composer (anchored
            to the composer's top via absolute + bottom:100%) so it
            can never overlap or push the composer. It doesn't take
            flex height because it's absolutely positioned. */}
        {hasMessages && (
          <div className="pointer-events-auto absolute bottom-full left-1/2 z-10 mb-2 -translate-x-1/2">
            <StatusBar
              phase={livePhase}
              elapsedMs={
                streaming ? lastAssistant?.elapsedMs : undefined
              }
              tokensIn={
                streaming ? lastAssistant?.tokensIn : undefined
              }
              tokensOut={
                streaming ? lastAssistant?.tokensOut : undefined
              }
              historyTokensIn={historyTokens.in}
              historyTokensOut={historyTokens.out}
              papersFound={streaming ? lastAssistant?.papersFound : undefined}
              modeId={(streaming ? lastAssistant?.mode : undefined) ?? mode}
              onCancel={streaming ? cancel : undefined}
            />
          </div>
        )}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
          className="relative rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] shadow-[var(--shadow-card)] focus-within:border-[var(--cnrs-blue)]"
        >
          {/* Input — full width. Token estimate is pinned bottom-right inside. */}
          <div className="relative">
            <textarea
              rows={2}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit();
                }
              }}
              placeholder="Ask the literature…  (Enter to send · Shift+Enter for newline)"
              // The parent <form> already shows a CNRS-blue focus-within
              // border around the composer; the global yellow focus-visible
              // outline (globals.css) would double-up and sit misaligned
              // with the rounded form, so suppress it for this textarea.
              className="w-full resize-none bg-transparent px-4 pt-3 pb-5 text-[15px] leading-relaxed outline-none focus-visible:[outline:none] placeholder:text-[var(--text-muted)]"
              disabled={streaming}
              autoFocus
            />
            {draft.trim().length > 0 && !streaming && (
              <span
                className="pointer-events-none absolute bottom-1 right-3 font-mono text-[10px] text-[var(--text-muted)]"
                title={`Estimated input tokens (~ chars / 4): ${estimateTokens(draft)}`}
              >
                ≈ {estimateTokens(draft).toLocaleString()} tok
              </span>
            )}
          </div>

          {/* Hybrid retrieval weight slider — only shown for modes that use it. */}
          {(mode === "advanced" || mode === "deep_research") && (
            <div className="flex items-center gap-2 border-t border-[var(--border)] px-3 py-1 text-[11px] text-[var(--text-muted)]">
              <span className="shrink-0">BM25</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={hybridWeight}
                onChange={(e) => setHybridWeight(parseFloat(e.target.value))}
                className="h-1 w-24 accent-[var(--cnrs-blue)]"
                title={`Retrieval: BM25 ${Math.round((1 - hybridWeight) * 100)}% / Vector ${Math.round(hybridWeight * 100)}%`}
                aria-label="Hybrid retrieval weight"
              />
              <span className="shrink-0">Vector</span>
              {hybridWeight !== 0.5 && (
                <button
                  type="button"
                  onClick={() => setHybridWeight(0.5)}
                  className="ml-1 rounded px-1 text-[10px] text-[var(--text-body)] hover:underline"
                  title="Reset to 50/50"
                >
                  reset
                </button>
              )}
              <span className="ml-auto font-mono text-[10px] tabular-nums">
                {Math.round((1 - hybridWeight) * 100)}/{Math.round(hybridWeight * 100)}
              </span>
            </div>
          )}

          {/* Toolbar: mode picker + DB favicons + Ask button on the same row. */}
          <div className="flex flex-wrap items-center gap-1.5 border-t border-[var(--border)] px-3 py-2">
            <CompactModePicker
              value={mode}
              onChange={setMode}
              disabled={streaming}
            />
            <KbPicker
              value={kbName}
              kbs={kbList}
              onChange={setKbName}
              disabled={streaming}
            />
            <DbFaviconRow
              selected={databases}
              onToggle={() => setShowDbPicker((s) => !s)}
              expanded={showDbPicker}
              disabled={streaming}
            />
            <TopNPill
              value={maxPapers}
              onChange={setMaxPapers}
              disabled={streaming}
            />
            <div className="ml-auto">
              {streaming ? (
                <button
                  type="button"
                  onClick={cancel}
                  className="rounded-[var(--radius-md)] border border-[var(--accent-fg)] px-3 py-1.5 text-xs font-medium text-[var(--accent-fg)] transition hover:bg-[var(--accent-fg)] hover:text-[var(--text-body)]"
                  title="Stop generation (Esc)"
                >
                  ◼ Stop
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!draft.trim()}
                  className={[
                    "inline-flex items-center gap-1.5 rounded-[var(--radius-md)] px-4 py-1.5 text-xs font-semibold transition",
                    accent.bg,
                    accent.text,
                    "disabled:cursor-not-allowed disabled:opacity-50",
                  ].join(" ")}
                  title={`Ask Perspicacité in ${currentMode.label} retrieval mode`}
                >
                  Ask Perspicacité
                  <span aria-hidden>→</span>
                </button>
              )}
            </div>
          </div>
          {showDbPicker && (
            <div className="relative border-t border-[var(--border)] p-3">
              <button
                type="button"
                onClick={() => setShowDbPicker(false)}
                title="Close database panel"
                aria-label="Close database panel"
                className="absolute right-2 top-2 grid h-6 w-6 place-items-center rounded text-[var(--text-muted)] opacity-60 transition hover:bg-[var(--surface-hover)] hover:text-[var(--text-body)] hover:opacity-100"
              >
                <svg
                  viewBox="0 0 24 24"
                  width="12"
                  height="12"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="M18 6 6 18M6 6l12 12" />
                </svg>
              </button>
              <DatabasePicker
                value={databases}
                onChange={setDatabases}
                disabled={streaming}
                compact
              />
            </div>
          )}
        </form>
        <p className="mt-2 text-center text-[10px] text-[var(--text-muted)]">
          Perspicacité may make mistakes. Verify against the linked sources.
        </p>
        </div>
      </div>
    </section>
  );
}

function DbFaviconRow({
  selected,
  onToggle,
  expanded,
  disabled,
}: {
  selected: DatabaseId[];
  onToggle: () => void;
  expanded: boolean;
  disabled?: boolean;
}) {
  const set = new Set(selected);
  const visible = DATABASES.filter((d) => set.has(d.id));
  const count = visible.length;

  // Avatar-stack layout: favicons overlap at 50% of their width
  // (negative margin) to keep the toolbar compact when many DBs
  // are selected. First item flows in at zero offset; each
  // subsequent gets -10px (≈ half its 20×20 visual box).
  const OVERLAP = -10;

  return (
    <button
      type="button"
      onClick={onToggle}
      disabled={disabled}
      aria-expanded={expanded}
      title={`${count}/${DATABASES.length} databases active — click to manage`}
      className="ml-1 inline-flex items-center rounded-[var(--radius-md)] border border-[var(--border)] px-2 py-1 transition hover:border-[var(--cnrs-blue)] disabled:cursor-not-allowed disabled:opacity-60"
    >
      {count === 0 ? (
        <span className="text-[11px] font-medium text-[var(--text-muted)]">
          No databases · pick some
        </span>
      ) : (
        <span className="flex items-center">
          {visible.map((d, i) => (
            <span
              key={d.id}
              style={{
                marginLeft: i === 0 ? 0 : OVERLAP,
                zIndex: visible.length - i,
              }}
              className="relative inline-flex"
            >
              <DatabaseGlyph id={d.id} size={14} />
            </span>
          ))}
          <span className="ml-1.5 text-[10px] font-mono text-[var(--text-muted)]">
            {count}/{DATABASES.length}
          </span>
        </span>
      )}
      <span className="sr-only">
        {count}/{DATABASES.length} databases active
      </span>
    </button>
  );
}

// Compact stepper for the per-query "Top-N retrieved papers" cap.
// Sits inline next to the database picker so the cap is visible at a
// glance and can be tweaked without opening Settings. Range 1–25 matches
// the backend RAGRequest.max_papers_retrieval bounds.
function TopNPill({
  value,
  onChange,
  disabled,
}: {
  value: number;
  onChange: (n: number) => void;
  disabled?: boolean;
}) {
  const clamp = (n: number) => Math.max(1, Math.min(25, Math.round(n)));
  return (
    <div
      title={`Cap on retrieved papers per query · current: ${value}. Click − / + to adjust (1–25).`}
      className="ml-1 inline-flex items-stretch overflow-hidden rounded-[var(--radius-md)] border border-[var(--border)] text-[11px] font-medium leading-none transition focus-within:border-[var(--cnrs-blue)] hover:border-[var(--cnrs-blue)]"
    >
      <button
        type="button"
        onClick={() => onChange(clamp(value - 1))}
        disabled={disabled || value <= 1}
        aria-label="Decrease top-N papers"
        className="grid w-6 place-items-center text-[var(--text-muted)] transition hover:bg-[var(--surface-hover)] hover:text-[var(--text-body)] disabled:cursor-not-allowed disabled:opacity-40"
      >
        −
      </button>
      <span className="grid place-items-center border-x border-[var(--border)] px-2 font-mono text-[var(--text-body)]">
        Top {value}
      </span>
      <button
        type="button"
        onClick={() => onChange(clamp(value + 1))}
        disabled={disabled || value >= 25}
        aria-label="Increase top-N papers"
        className="grid w-6 place-items-center text-[var(--text-muted)] transition hover:bg-[var(--surface-hover)] hover:text-[var(--text-body)] disabled:cursor-not-allowed disabled:opacity-40"
      >
        +
      </button>
    </div>
  );
}

// Compact corpus selector for the composer toolbar. Initialised from
// Settings → "Default knowledge base" but overridable per conversation
// (like the mode / databases / Top-N toggles beside it). Values:
//   ""     → no KB (web / databases only)
//   "auto" → backend routes to the best-matching corpus by similarity
//   <name> → that knowledge base
function KbPicker({
  value,
  kbs,
  onChange,
  disabled,
}: {
  value: string | null;
  kbs: KBSummary[];
  onChange: (v: string | null) => void;
  disabled?: boolean;
}) {
  return (
    <label
      title="Corpus to search · 'No KB' = web/databases only · 'Auto' = best-matching corpus by similarity. Overrides the Settings default for this conversation."
      className="ml-1 inline-flex items-center gap-1 rounded-[var(--radius-md)] border border-[var(--border)] pl-2 leading-none transition focus-within:border-[var(--cnrs-blue)] hover:border-[var(--cnrs-blue)]"
    >
      <svg
        viewBox="0 0 24 24"
        width="12"
        height="12"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden
        className="shrink-0 text-[var(--text-muted)]"
      >
        <ellipse cx="12" cy="5" rx="8" ry="3" />
        <path d="M4 5v6c0 1.6 3.6 3 8 3s8-1.4 8-3V5" />
        <path d="M4 11v6c0 1.6 3.6 3 8 3s8-1.4 8-3v-6" />
      </svg>
      <select
        value={value ?? ""}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value === "" ? null : e.target.value)}
        aria-label="Knowledge base to search"
        className="max-w-[11rem] cursor-pointer truncate bg-transparent py-1.5 pr-2 text-[11px] font-medium text-[var(--text-body)] outline-none disabled:cursor-not-allowed disabled:opacity-50"
      >
        <option value="">No KB · web only</option>
        <option value="auto">Auto · best match</option>
        {kbs.length > 0 && <option disabled>──────────</option>}
        {kbs.map((k) => (
          <option key={k.name} value={k.name}>
            {k.name}
            {typeof k.paper_count === "number" ? ` (${k.paper_count})` : ""}
          </option>
        ))}
      </select>
    </label>
  );
}

function CompactModePicker({
  value,
  onChange,
  disabled,
}: {
  value: RAGMode;
  onChange: (m: RAGMode) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-wrap gap-1" role="radiogroup" aria-label="Mode">
      {MODES.map((m) => {
        const selected = value === m.id;
        const accent = accentClasses(m.accent);
        return (
          <button
            key={m.id}
            type="button"
            role="radio"
            aria-checked={selected}
            disabled={disabled}
            onClick={() => onChange(m.id)}
            title={`${m.label} · ${m.blurb} · ~${m.latency}`}
            className={[
              "rounded-full px-2.5 py-1 text-[11px] font-medium transition",
              selected
                ? `${accent.bg} ${accent.text} shadow-sm`
                : "border border-[var(--border)] bg-transparent text-[var(--text-muted)] hover:border-[var(--cnrs-blue)] hover:text-[var(--text-body)]",
              disabled && "cursor-not-allowed opacity-60",
            ].join(" ")}
          >
            {m.label}
          </button>
        );
      })}
    </div>
  );
}

function HeroEmptyState({
  mode,
  onPick,
}: {
  mode: RAGMode;
  onPick: (prompt: string) => void;
}) {
  const m = MODES.find((x) => x.id === mode)!;
  return (
    <div className="flex flex-col items-center justify-center gap-6 pt-12 pb-6 text-center">
      <span
        className="grid h-14 w-14 place-items-center rounded-full"
        style={{ background: "var(--cnrs-yellow)" }}
        aria-hidden
      />
      <div>
        <h1 className="text-3xl font-semibold tracking-tight text-[var(--accent-fg)] md:text-4xl">
          Ask the literature.
        </h1>
        <p className="mt-2 text-sm text-[var(--text-muted)]">
          {m.label} mode · {m.blurb}
        </p>
      </div>
      <div className="mt-2 grid w-full max-w-2xl grid-cols-1 gap-2 sm:grid-cols-2">
        {EXAMPLE_PROMPTS.map((p) => (
          <button
            key={p}
            type="button"
            onClick={() => onPick(p)}
            className="rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2.5 text-left text-sm text-[var(--text-body)] transition hover:border-[var(--cnrs-blue)] hover:shadow-[var(--shadow-card)]"
          >
            {p}
          </button>
        ))}
      </div>
    </div>
  );
}

function UserMessage({ turn }: { turn: Turn }) {
  return (
    <li className="flex justify-end">
      <div className="max-w-[85%] rounded-[var(--radius-lg)] bg-[var(--cnrs-blue)] px-4 py-2.5 text-[15px] leading-relaxed text-white">
        {turn.text}
      </div>
    </li>
  );
}

function AssistantMessage({
  turn,
  streaming,
}: {
  turn: Turn;
  streaming: boolean;
}) {
  const m = turn.mode ? MODES.find((x) => x.id === turn.mode) : undefined;
  const accent = m ? accentClasses(m.accent) : undefined;
  // Typewriter: pace text into the view at ~12 chars/frame (~720
  // chars/s ≈ 180 tokens/s) while streaming. The backend often
  // returns the full answer in one burst, so the typewriter is what
  // sells "generation in progress" to the eye. Snap to full text
  // once the turn is final.
  const displayed = useTypewriter(
    turn.text,
    // While streaming, type at a deliberate pace. Once the stream ends, finish
    // the buffered tail (e.g. the references block, which arrives in one chunk)
    // at a fast but FINITE rate so it eases in smoothly instead of snapping in
    // all at once. Historical loads stay instant — useTypewriter starts caught
    // up, so no animation runs.
    turn.streaming && streaming ? 12 : 48,
  );

  // Progressively reveal steps + sources while streaming, so a burst
  // of backend events doesn't paint as one block. Once the turn is
  // final we show everything immediately. Delays are deliberately
  // generous (≥ 400 ms) so the cascade reads as motion, not a stutter.
  const allSteps = turn.steps ?? [];
  const allSources = turn.sources ?? [];
  const stagger = turn.streaming && streaming;
  // Pace: 300 ms / step, 200 ms / source. Fast enough to feel like
  // each event has its own moment, slow enough to be perceptible.
  const displayedSteps = useStaggeredList(allSteps, stagger ? 300 : 0);
  const displayedSources = useStaggeredList(allSources, stagger ? 200 : 0);

  return (
    <li className="flex flex-col gap-3">
      {/* Mode tag + final stats — pill colour mirrors the mode button in the composer */}
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <span
            className={[
              "rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
              accent?.bg ?? "bg-[var(--cnrs-grey-light)]",
              accent?.text ?? "text-[var(--cnrs-blue)]",
            ].join(" ")}
            title={m ? `${m.label} mode · ${m.blurb}` : undefined}
          >
            {m?.label ?? "assistant"}
          </span>
          {turn.phase === "cancelled" && (
            <span className="font-mono text-[10px] text-[var(--text-muted)]">
              cancelled
            </span>
          )}
        </div>
        {/* Helper text — short user-facing explanation of what this mode
            does and roughly how long it'll take. Only shown while the
            turn is streaming, so finished answers don't carry the noise. */}
        {m && turn.streaming && (
          <p className="text-[11px] leading-snug text-[var(--text-muted)]">
            {m.helper}
          </p>
        )}
      </div>

      {/* Agentic trail — show the planned phases for the mode upfront,
          and update progressively as backend events arrive. Open by default. */}
      {turn.mode && (
        <ThinkingSteps
          steps={displayedSteps}
          sources={displayedSources}
          defaultOpen={true}
          running={!!(turn.streaming && streaming)}
          errored={!!turn.error}
          modeId={turn.mode}
        />
      )}

      {/* Deep Research diagnostic badge — cycles + papers + completion reason.
          Only shown for completed deep_research turns where metadata arrived. */}
      {turn.iterationCount !== undefined && turn.mode === "deep_research" && (
        <div className="mt-1 flex items-center gap-2 text-[11px] text-[var(--text-muted)]">
          <span className="font-mono">
            {turn.iterationCount} cycle{turn.iterationCount !== 1 ? "s" : ""}
          </span>
          {turn.diagnostic?.papers_retrieved !== undefined && (
            <>
              <span aria-hidden>·</span>
              <span className="font-mono">{turn.diagnostic.papers_retrieved as number} papers</span>
            </>
          )}
          {turn.completionReason && turn.completionReason !== "complete" && (
            <>
              <span aria-hidden>·</span>
              <span className="italic">{turn.completionReason.replace(/_/g, " ")}</span>
            </>
          )}
        </div>
      )}

      {/* Sources first (Perplexity pattern). The grid uses .cnrs-stagger
          so each new card fades in with a small per-index delay; combined
          with `useStaggeredList` above, this turns a single-frame burst
          into a visible cascade. */}
      {displayedSources.length > 0 && (
        <div>
          <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
            Sources · {displayedSources.length}
            {allSources.length > displayedSources.length && (
              <span className="ml-1 text-[var(--text-body)]">
                / {allSources.length}
              </span>
            )}
          </p>
          <div className="cnrs-stagger grid grid-cols-1 gap-1.5 sm:grid-cols-2">
            {displayedSources.map((s, i) => (
              <div key={i} style={{ ["--i" as string]: i }}>
                <SourcePill source={s} index={i} />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Answer */}
      <div className="text-[15px] leading-relaxed text-[var(--text-body)]">
        {turn.error ? (
          <p className="rounded-[var(--radius-md)] border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            ⚠ {turn.error}
          </p>
        ) : displayed ? (
          <>
            <Markdown>{displayed}</Markdown>
            {turn.streaming && streaming && displayed.length < turn.text.length && (
              <span className="cnrs-caret" aria-hidden />
            )}
          </>
        ) : turn.streaming && streaming ? (
          <span className="inline-flex items-center gap-2 text-sm text-[var(--text-muted)]">
            <span className="cnrs-sun" aria-hidden />
            <span>Working</span>
            <span className="cnrs-typing-dots text-[var(--accent-fg)]" aria-hidden>
              <span /><span /><span />
            </span>
          </span>
        ) : (
          <p className="text-sm text-[var(--text-muted)]">(no answer)</p>
        )}
      </div>

      {/* Final summary footer — appears once the turn is complete.
          Total elapsed + token usage + (estimated) cost in USD. */}
      {!turn.streaming && (turn.elapsedMs || turn.tokensOut || turn.tokensIn) && (
        <TurnSummary turn={turn} />
      )}
    </li>
  );
}

function TurnSummary({ turn }: { turn: Turn }) {
  const model = useLlmModelLabel();
  const tokensIn = turn.tokensIn ?? 0;
  const tokensOut = turn.tokensOut ?? 0;
  const cost = estimateCostUsd(model, tokensIn, tokensOut);
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-[var(--radius-md)] border border-dashed border-[var(--border)] bg-[var(--bg-soft)] px-3 py-1.5 text-[10px] font-mono text-[var(--text-muted)]">
      {turn.elapsedMs ? (
        <span title="Elapsed time">
          {(turn.elapsedMs / 1000).toFixed(1)}s
        </span>
      ) : null}
      <span title="Input tokens · Output tokens">
        {tokensIn.toLocaleString()} <span aria-hidden>↑</span>{" "}
        {tokensOut.toLocaleString()} <span aria-hidden>↓</span>
      </span>
      {cost !== null && (
        <span title="Estimated cost (USD) based on the running model's public price per token">
          ≈ {formatUsd(cost)}
        </span>
      )}
      {model && (
        <span className="ml-auto" title={model}>
          {model.split("/").slice(-1)[0]}
        </span>
      )}
    </div>
  );
}
