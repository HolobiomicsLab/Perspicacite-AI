"use client";

import { useEffect, useState } from "react";
import type { ChatSource, ThinkingStep } from "@/lib/chat";
import { DATABASES, describeProvider } from "@/lib/databases";
import { DatabaseGlyph } from "./DatabaseGlyph";
import { JournalFavicon } from "./JournalFavicon";
import { PhaseGlyph } from "./PhaseGlyph";
import { SourceDrawer } from "./SourceDrawer";
import {
  MODES,
  type ModePhase,
  type RAGMode,
} from "@/lib/modes";

// Agentic trail — Claude-Code-style planning:
//   - The expected sequence (phases for the chosen mode) is rendered
//     immediately as a top-level list with empty checkboxes.
//   - As backend events arrive, each step is matched into a phase
//     and rendered as an indented sub-row under it (with its own
//     vertical spine, so nesting reads like a Python function).
//   - The phase containing the most recent step is "running" and
//     pulses yellow; earlier phases mark "done" with a check; later
//     phases stay "planned" and faded.
//   - Sources arriving via meta events are rendered as compact
//     one-line mini-cards under the retrieval phase.

export function ThinkingSteps({
  steps,
  sources,
  defaultOpen = true,
  running = false,
  modeId,
}: {
  steps: ThinkingStep[];
  sources?: ChatSource[];
  defaultOpen?: boolean;
  running?: boolean;
  modeId?: RAGMode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  useEffect(() => {
    if (running && steps.length > 0) setOpen(true);
  }, [running, steps.length]);

  const mode = modeId ? MODES.find((m) => m.id === modeId) : undefined;
  const phases = mode?.phases ?? [];
  const collapsedSteps = collapseBatch(steps);
  const showPhases = phases.length > 0;
  if (!showPhases && collapsedSteps.length === 0) return null;

  const buckets = bucketSteps(collapsedSteps, phases);
  const activePhaseIdx = findActivePhase(buckets, phases);
  // Decide which phase shows retrieved sources — the first phase whose
  // id implies retrieval. Falls back to the active phase.
  const sourcesPhaseIdx = phases.findIndex((p) =>
    /retriev|search|collect|tools/.test(p.id),
  );
  const totalSteps = collapsedSteps.length;

  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--bg-soft)]">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-3 py-1.5 text-left text-xs text-[var(--text-muted)] hover:text-[var(--accent-fg)]"
        aria-expanded={open}
      >
        <span className="flex items-center gap-2">
          <span aria-hidden>{open ? "▾" : "▸"}</span>
          <span>
            Agentic trail
            {totalSteps > 0
              ? ` · ${totalSteps} event${totalSteps === 1 ? "" : "s"}`
              : ""}
            {sources && sources.length > 0
              ? ` · ${sources.length} source${sources.length === 1 ? "" : "s"}`
              : ""}
          </span>
          {running && (
            <span
              aria-hidden
              className="cnrs-typing-dots text-[var(--accent-fg)]"
            >
              <span />
              <span />
              <span />
            </span>
          )}
        </span>
        <span className="font-mono text-[10px] text-[var(--text-muted)]">
          {open ? "hide" : "show"}
        </span>
      </button>
      {open && (
        <ol className="cnrs-step-spine cnrs-stagger flex flex-col gap-2 border-t border-[var(--border)] px-3 py-3 pl-4">
          {showPhases
            ? phases.map((phase, i) => {
                const phaseSteps = buckets.byPhase[phase.id] ?? [];
                const status: PhaseStatus =
                  i < activePhaseIdx
                    ? "done"
                    : i === activePhaseIdx
                      ? running
                        ? "running"
                        : "done"
                      : "planned";
                const phaseSources = i === sourcesPhaseIdx ? sources ?? [] : [];
                return (
                  <li
                    key={phase.id}
                    style={{ ["--i" as string]: i }}
                    className="relative"
                  >
                    <PhaseRow
                      phase={phase}
                      status={status}
                      steps={phaseSteps}
                      sources={phaseSources}
                    />
                  </li>
                );
              })
            : null}
          {!showPhases &&
            collapsedSteps.map((s, i) => (
              <li
                key={s.id}
                style={{ ["--i" as string]: i }}
                className="relative"
              >
                <StepRow
                  step={s}
                  running={running && i === collapsedSteps.length - 1}
                />
              </li>
            ))}
        </ol>
      )}
    </div>
  );
}

type PhaseStatus = "planned" | "running" | "done";

type Buckets = {
  byPhase: Record<string, ThinkingStep[]>;
};

// Chronological bucketing: walk steps in order, maintaining the
// "currently-active phase index". A step that matches a phase
// updates the active phase and lands there; a step that matches no
// phase still lands in the currently-active one (so it appears in
// place, instead of being banished to an "Other events" bin at the
// bottom of the trail).
function bucketSteps(
  steps: ThinkingStep[],
  phases: ModePhase[],
): Buckets {
  const byPhase: Record<string, ThinkingStep[]> = Object.fromEntries(
    phases.map((p) => [p.id, [] as ThinkingStep[]]),
  );
  if (phases.length === 0) return { byPhase };
  let activeIdx = 0;
  for (const s of steps) {
    const blob = stepSearchBlob(s).toLowerCase();
    const idx = phases.findIndex((p) => p.match.some((m) => blob.includes(m)));
    if (idx >= 0) activeIdx = idx;
    byPhase[phases[activeIdx].id].push(s);
  }
  return { byPhase };
}

function stepSearchBlob(s: ThinkingStep): string {
  const parts: string[] = [s.label, s.kind];
  if (s.detail) {
    if (s.detail.stage) parts.push(s.detail.stage);
    if (s.detail.phase) parts.push(s.detail.phase);
    if (s.detail.rephrased) parts.push(s.detail.rephrased);
    if (s.detail.original) parts.push(s.detail.original);
    if (s.detail.providers) parts.push(...s.detail.providers);
  }
  return parts.join(" ");
}

function findActivePhase(buckets: Buckets, phases: ModePhase[]): number {
  for (let i = phases.length - 1; i >= 0; i--) {
    if ((buckets.byPhase[phases[i].id] ?? []).length > 0) return i;
  }
  return 0;
}

function collapseBatch(steps: ThinkingStep[]): ThinkingStep[] {
  const out: ThinkingStep[] = [];
  for (const s of steps) {
    const last = out[out.length - 1];
    if (
      s.kind === "batch_progress" &&
      last?.kind === "batch_progress" &&
      last.detail?.stage === s.detail?.stage
    ) {
      out[out.length - 1] = s;
    } else {
      out.push(s);
    }
  }
  return out;
}

function PhaseRow({
  phase,
  status,
  steps,
  sources,
}: {
  phase: ModePhase;
  status: PhaseStatus;
  steps: ThinkingStep[];
  sources: ChatSource[];
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2 text-[13px]">
        <PhaseIcon status={status} />
        {phase.glyph && (
          <span
            className={[
              "inline-flex shrink-0 items-center",
              status === "done" && "text-[var(--accent-fg)]",
              status === "running" && "text-[var(--accent-fg)] cnrs-blink",
              status === "planned" && "text-[var(--text-muted)] opacity-50",
            ]
              .filter(Boolean)
              .join(" ")}
            aria-hidden
          >
            <PhaseGlyph glyph={phase.glyph} size={14} />
          </span>
        )}
        <span
          className={[
            "font-medium",
            status === "done" && "text-[var(--text-body)]",
            status === "running" && "text-[var(--accent-fg)] font-semibold",
            status === "planned" && "text-[var(--text-muted)]",
          ]
            .filter(Boolean)
            .join(" ")}
        >
          {phase.label}
        </span>
        {phase.hint && status === "planned" && (
          <span className="text-[10px] text-[var(--text-muted)]">
            ({phase.hint})
          </span>
        )}
        {status === "running" && (
          <span
            aria-hidden
            className="cnrs-typing-dots text-[var(--cnrs-yellow)]"
          >
            <span />
            <span />
            <span />
          </span>
        )}
      </div>

      {/* Indented sub-spine for the steps + sources that landed in this phase. */}
      {(steps.length > 0 || sources.length > 0) && (
        <ol className="cnrs-step-spine cnrs-stagger ml-3 flex flex-col gap-1.5 pl-4">
          {steps.map((s, i) => (
            <li
              key={s.id}
              className="relative"
              style={{ ["--i" as string]: i }}
            >
              <StepRow step={s} running={false} />
            </li>
          ))}
          {sources.map((s, i) => (
            <li
              key={`src-${i}`}
              className="relative"
              style={{ ["--i" as string]: steps.length + i }}
            >
              <MiniSourceCard source={s} index={i} />
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function PhaseIcon({ status }: { status: PhaseStatus }) {
  if (status === "done") {
    return (
      <span
        aria-hidden
        className="relative z-10 grid h-4 w-4 shrink-0 place-items-center rounded-full bg-[var(--cnrs-green)] text-[10px] text-[var(--cnrs-blue)] ring-1 ring-[var(--cnrs-green)]"
      >
        ✓
      </span>
    );
  }
  if (status === "running") {
    return (
      <span
        aria-hidden
        className="relative z-10 grid h-4 w-4 shrink-0 place-items-center rounded-full bg-[var(--cnrs-yellow)] text-[10px] text-[var(--cnrs-blue)] ring-2 ring-[var(--cnrs-yellow)]/40"
        style={{ animation: "cnrs-sun-pulse 1.6s ease-in-out infinite" }}
      >
        ▶
      </span>
    );
  }
  return (
    <span
      aria-hidden
      className="relative z-10 grid h-4 w-4 shrink-0 place-items-center rounded-full border border-dashed border-[var(--cnrs-grey)] text-[10px] text-[var(--text-muted)]"
    >
      ○
    </span>
  );
}

function StepRow({
  step,
  running,
}: {
  step: ThinkingStep;
  running: boolean;
}) {
  if (step.kind === "query_rephrased" && step.detail) {
    return (
      <Row icon="↻" running={running}>
        <p className="font-medium text-[var(--text-body)]">Query refined</p>
        {step.detail.original && (
          <p className="text-[11px] text-[var(--text-muted)] line-through">
            {step.detail.original}
          </p>
        )}
        {step.detail.rephrased && (
          <p className="text-[12px] text-[var(--accent-fg)]">
            {step.detail.rephrased}
          </p>
        )}
      </Row>
    );
  }

  if (step.kind === "provider_progress" && step.detail) {
    const phase = step.detail.phase ?? "start";
    const providers = step.detail.providers ?? [];
    const byProvider = step.detail.byProvider ?? {};
    return (
      <Row icon={phase === "start" ? "⌕" : "✓"} running={running}>
        <p className="font-medium text-[var(--text-body)]">
          {phase === "start" ? "Querying databases" : "Database results"}
        </p>
        {phase === "start" && providers.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1">
            {providers.map((p) => (
              <ProviderPill key={p} name={p} />
            ))}
          </div>
        )}
        {phase === "done" && Object.keys(byProvider).length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1">
            {Object.entries(byProvider)
              .sort((a, b) => b[1] - a[1])
              .map(([p, count]) => (
                <ProviderPill key={p} name={p} count={count} />
              ))}
          </div>
        )}
      </Row>
    );
  }

  if (step.kind === "batch_progress" && step.detail) {
    const { current = 0, total = 0 } = step.detail;
    const pct = total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 0;
    const done = total > 0 && current >= total;
    return (
      <Row icon={done ? "✓" : "⋯"} running={running && !done}>
        <div className="flex items-baseline gap-2">
          <p className="font-medium text-[var(--text-body)]">{step.label}</p>
          <span className="font-mono text-[10px] tabular-nums text-[var(--text-muted)]">
            {current}/{total}
          </span>
        </div>
        {total > 0 && (
          <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-[var(--cnrs-grey-light)]">
            <div
              className="h-full transition-all duration-300"
              style={{
                width: `${pct}%`,
                background: done ? "var(--cnrs-green)" : "var(--cnrs-yellow)",
              }}
            />
          </div>
        )}
      </Row>
    );
  }

  // status (plain message) — auto-detect provider names and inline
  // favicons, and surface an actionable hint when retrieval came up
  // empty (a common pattern is short queries that miss the literature's
  // canonical multi-word term).
  const lower = step.label.toLowerCase();
  const isZeroResults =
    /no relevant papers|0 hits|no papers found|no results/.test(lower);
  return (
    <Row icon={isZeroResults ? "∅" : "·"} running={running}>
      <StatusMessageInline text={step.label} />
      {isZeroResults && (
        <p className="mt-1 rounded-[var(--radius-sm)] border border-[var(--cnrs-yellow)] bg-[var(--cnrs-yellow)]/30 px-2 py-1 text-[11px] leading-snug text-[var(--text-body)]">
          <span aria-hidden>💡 </span>
          Try a more specific phrase — many techniques use a longer
          canonical name (e.g. <em>ion identity</em>{" "}
          <strong>molecular</strong> <em>networking</em>) or a
          well-known acronym. Adding the missing qualifier usually
          unlocks hits.
        </p>
      )}
    </Row>
  );
}

function Row({
  icon,
  running,
  children,
}: {
  icon: string;
  running: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-start gap-2 text-xs">
      <span
        aria-hidden
        className={[
          "relative z-10 grid h-4 w-4 shrink-0 place-items-center rounded-full text-[10px]",
          running
            ? "bg-[var(--cnrs-yellow)] text-[var(--cnrs-blue)] ring-2 ring-[var(--cnrs-yellow)]/40"
            : "bg-[var(--surface)] text-[var(--text-body)] ring-1 ring-[var(--border)]",
        ].join(" ")}
        style={
          running
            ? { animation: "cnrs-sun-pulse 1.6s ease-in-out infinite" }
            : undefined
        }
      >
        {icon}
      </span>
      <div className="min-w-0 flex-1 pt-px">{children}</div>
    </div>
  );
}

// Compact one-line card surfaced inside the trail: glyph + truncated
// title + first author + year + journal favicon. Clicking opens the
// full source drawer.
function MiniSourceCard({
  source,
  index,
}: {
  source: ChatSource;
  index: number;
}) {
  const [open, setOpen] = useState(false);
  const title = source.title ?? source.doi ?? `Source ${index + 1}`;
  const author =
    source.authors?.length
      ? source.authors.length > 1
        ? `${source.authors[0]} et al.`
        : source.authors[0]
      : undefined;
  const providers = Array.from(
    new Set(
      [
        source.provider,
        ...(source.providers ?? []),
        ...(source.discovery_sources ?? []),
      ].filter((s): s is string => !!s),
    ),
  );

  return (
    <>
      <div
        className="group flex items-stretch rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] transition hover:border-[var(--cnrs-blue)]/40"
        title={`Open full detail · ${title}`}
      >
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="flex min-w-0 flex-1 flex-col gap-0.5 px-2 py-1.5 text-left text-[11px]"
        >
          <p className="line-clamp-2 leading-snug text-[var(--accent-fg)] group-hover:underline">
            {title}
          </p>
          <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-[10px] text-[var(--text-muted)]">
            {author && (
              <span>
                {author}
                {typeof source.year === "number" ? ` · ${source.year}` : ""}
              </span>
            )}
            {source.journal && (
              <span className="inline-flex items-center gap-1 italic">
                <JournalFavicon name={source.journal} size={9} />
                {source.journal}
              </span>
            )}
            {providers.length > 0 && (
              <span className="inline-flex items-center">
                {providers.slice(0, 4).map((p, i) => (
                  <span
                    key={p}
                    style={{
                      marginLeft: i === 0 ? 0 : -5,
                      zIndex: providers.length - i,
                    }}
                    className="relative inline-flex"
                  >
                    <DatabaseGlyph id={p} size={9} />
                  </span>
                ))}
              </span>
            )}
          </div>
        </button>
        {/* Right-side affordance — mirrors the full source pill. */}
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="flex shrink-0 items-center border-l border-[var(--border)] px-2 text-[10px] text-[var(--text-muted)] transition hover:bg-[var(--surface-hover)] hover:text-[var(--text-body)]"
          aria-label="Open abstract in side panel"
          title="Open abstract in side panel"
        >
          <span aria-hidden>↗</span>
        </button>
      </div>
      {open && (
        <SourceDrawer source={source} onClose={() => setOpen(false)} />
      )}
    </>
  );
}

// Parse a status message and replace any database name mention with
// its favicon glyph, inline. Conservative — only swaps when we find
// an exact label / id match against the canonical DATABASES list.
function StatusMessageInline({ text }: { text: string }) {
  // Build a list of patterns (label + lowercased id forms) once.
  const patterns: Array<{ re: RegExp; id: string }> = DATABASES.flatMap(
    (d) => {
      const variants = new Set<string>([
        d.label,
        d.label.toLowerCase(),
        d.id.replace(/_/g, " "),
        d.id.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
      ]);
      return [...variants].map((v) => ({
        // Word-boundary match; escape regex metachars.
        re: new RegExp(`\\b${v.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "gi"),
        id: d.id,
      }));
    },
  );

  // Walk the text, finding the earliest match in each iteration. We
  // emit plain text segments alternating with provider chips.
  type Token = { kind: "text"; value: string } | { kind: "db"; id: string; label: string };
  const tokens: Token[] = [];
  let remaining = text;
  // Hard cap to avoid pathological loops on very long messages.
  for (let i = 0; i < 50 && remaining; i++) {
    let earliest: { index: number; length: number; id: string; label: string } | null = null;
    for (const { re, id } of patterns) {
      re.lastIndex = 0;
      const m = re.exec(remaining);
      if (m && (earliest === null || m.index < earliest.index)) {
        earliest = { index: m.index, length: m[0].length, id, label: m[0] };
      }
    }
    if (!earliest) {
      tokens.push({ kind: "text", value: remaining });
      break;
    }
    if (earliest.index > 0) {
      tokens.push({ kind: "text", value: remaining.slice(0, earliest.index) });
    }
    tokens.push({ kind: "db", id: earliest.id, label: earliest.label });
    remaining = remaining.slice(earliest.index + earliest.length);
  }

  // Collapse adjacent "across <db>, <db>, <db>…" sequences into a
  // tight cluster of favicons by trimming punctuation between two
  // consecutive db tokens.
  const cleaned: typeof tokens = [];
  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    if (tok.kind === "text" && i > 0 && tokens[i - 1].kind === "db") {
      // Strip leading ", " or " ," and a trailing "…"/"..." that
      // might immediately follow a db chip.
      let v = tok.value;
      // If the *next* token is also a db, drop separators between them.
      if (i + 1 < tokens.length && tokens[i + 1].kind === "db") {
        v = v.replace(/^[\s,]+/, " ").replace(/[\s,]+$/, " ");
        if (v.trim() === "") continue;
      } else {
        // Last db chip: drop trailing "…" / "..."
        v = v.replace(/^[\s,]*/, "").replace(/[\s…]*\.{0,3}\s*$/, "");
        if (v.trim() === "") continue;
        v = " " + v;
      }
      cleaned.push({ kind: "text", value: v });
    } else {
      cleaned.push(tok);
    }
  }

  return (
    <p className="flex flex-wrap items-center gap-1 text-[var(--text-body)]">
      {cleaned.map((tok, i) =>
        tok.kind === "text" ? (
          <span key={i}>{tok.value}</span>
        ) : (
          <span key={i} title={tok.label} className="inline-flex">
            <DatabaseGlyph id={tok.id} size={10} />
          </span>
        ),
      )}
    </p>
  );
}

export function ProviderPill({
  name,
  count,
}: {
  name: string;
  count?: number;
  /** Deprecated: external homepage link kept off by default — clicking
   *  a provider pill in the trail shouldn't ferry the user off the app. */
  href?: string;
}) {
  const d = describeProvider(name);
  const label = d?.label ?? name;
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full border border-[var(--border)] bg-[var(--surface)] px-1.5 py-0.5"
      title={label}
    >
      <DatabaseGlyph id={d?.id ?? name} size={10} />
      <span className="text-[10px] font-medium text-[var(--text-body)]">
        {label}
      </span>
      {typeof count === "number" && (
        <span className="font-mono text-[10px] text-[var(--text-muted)]">
          {count}
        </span>
      )}
    </span>
  );
}
