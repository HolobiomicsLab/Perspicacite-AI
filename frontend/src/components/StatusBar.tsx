"use client";

import { MODES, accentClasses, type RAGMode } from "@/lib/modes";
import { useAnimatedNumber } from "@/lib/streamAnimations";
import { useLlmModelLabel } from "./useLlmModelLabel";

export type ChatPhase =
  | "idle"
  | "sending"
  | "retrieving"
  | "synthesizing"
  | "done"
  | "error"
  | "cancelled";

const PHASE_LABEL: Record<ChatPhase, string> = {
  idle: "Idle",
  sending: "Sending query",
  retrieving: "Retrieving sources",
  synthesizing: "Generating answer",
  done: "Done",
  error: "Error",
  cancelled: "Cancelled",
};

export function StatusBar({
  phase,
  label,
  elapsedMs,
  tokensIn,
  tokensOut,
  historyTokensIn = 0,
  historyTokensOut = 0,
  papersFound,
  modeId,
  onCancel,
}: {
  phase: ChatPhase;
  label?: string;
  elapsedMs?: number;
  /** Current-turn input tokens. Undefined when no turn is in flight. */
  tokensIn?: number;
  /** Current-turn output tokens. */
  tokensOut?: number;
  /** Cumulative input tokens from past assistant turns in this conversation. */
  historyTokensIn?: number;
  /** Cumulative output tokens from past assistant turns. */
  historyTokensOut?: number;
  papersFound?: number;
  modeId?: RAGMode;
  onCancel?: () => void;
}) {
  const elapsed = ((elapsedMs ?? 0) / 1000).toFixed(1);
  const running =
    phase === "sending" || phase === "retrieving" || phase === "synthesizing";
  const display = label ?? PHASE_LABEL[phase];
  const llm = useLlmModelLabel();
  const mode = modeId ? MODES.find((m) => m.id === modeId) : undefined;
  const modeAccent = mode ? accentClasses(mode.accent) : undefined;

  // Animate the counters so a jump from 0→N feels like motion, even
  // when the underlying LLM burst-emits its whole answer.
  const animTokensIn = useAnimatedNumber(tokensIn ?? 0);
  const animTokensOut = useAnimatedNumber(tokensOut ?? 0);

  // Don't render anything in idle to keep the chat surface clean.
  if (phase === "idle" && !label) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-full border border-[var(--border)] bg-[var(--surface)] px-3.5 py-1.5 text-xs text-[var(--text-muted)] shadow-[var(--shadow-card)]"
    >
      {/* Heartbeat — yellow sun pulse when working, static when idle/done.
          Wider gap so the pulse halo has visual room and the text
          doesn't sit on top of it. */}
      <span className="flex items-center gap-2.5">
        {running ? (
          <span className="cnrs-sun" aria-hidden />
        ) : phase === "error" ? (
          <span
            aria-hidden
            className="h-2 w-2 rounded-full"
            style={{ background: "#dc2626" }}
          />
        ) : (
          <span
            aria-hidden
            className="h-2 w-2 rounded-full"
            style={{ background: "var(--cnrs-grey)" }}
          />
        )}
        <span
          className={
            phase === "error"
              ? "font-medium text-red-700"
              : "font-medium text-[var(--text-body)]"
          }
        >
          {display}
        </span>
        {running && (
          <span
            className="cnrs-typing-dots text-[var(--accent-fg)]"
            aria-hidden
          >
            <span />
            <span />
            <span />
          </span>
        )}
      </span>

      {(elapsedMs != null && elapsedMs > 0) && (
        <>
          <Sep />
          <span className="font-mono tabular-nums">{elapsed}s</span>
        </>
      )}

      {typeof papersFound === "number" && papersFound > 0 && (
        <>
          <Sep />
          <span>{papersFound} papers</span>
        </>
      )}

      {(tokensIn || tokensOut || running || historyTokensIn || historyTokensOut) && (
        <>
          <Sep />
          <span
            className="font-mono tabular-nums"
            title={
              `Conversation so far — in: ${historyTokensIn} · out: ${historyTokensOut}` +
              (running || tokensIn || tokensOut
                ? `\nCurrent turn — in: ${tokensIn ?? 0} · out: ${tokensOut ?? 0}`
                : "")
            }
          >
            {(historyTokensIn > 0 || historyTokensOut > 0) && (
              <span className="text-[var(--text-muted)]">
                {historyTokensIn.toLocaleString()}{" "}
                <span aria-hidden>↑</span>{" "}
                {historyTokensOut.toLocaleString()}{" "}
                <span aria-hidden>↓</span>
              </span>
            )}
            {/* Only show the "+ current" segment when there's an
                actual in-flight turn with non-zero counters. */}
            {(running || (tokensIn ?? 0) > 0 || (tokensOut ?? 0) > 0) && (
              <>
                {(historyTokensIn > 0 || historyTokensOut > 0) && (
                  <span aria-hidden className="text-[var(--text-muted)]">
                    {" + "}
                  </span>
                )}
                <span
                  className={
                    running
                      ? "text-[var(--accent-fg)] font-semibold"
                      : "text-[var(--text-body)]"
                  }
                >
                  {animTokensIn.toLocaleString()}{" "}
                  <span aria-hidden>↑</span>{" "}
                  {animTokensOut.toLocaleString()}{" "}
                  <span aria-hidden>↓</span>
                </span>
                {running && (tokensOut ?? 0) === 0 && (
                  <span
                    aria-hidden
                    className="ml-1 inline-block h-1 w-6 rounded-full cnrs-shimmer"
                    title="Waiting for first token"
                  />
                )}
              </>
            )}
          </span>
        </>
      )}

      {mode && (
        <>
          <Sep />
          <span
            className={[
              "inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
              modeAccent?.bg ?? "bg-[var(--cnrs-grey-light)]",
              modeAccent?.text ?? "text-[var(--text-body)]",
            ].join(" ")}
            title={`${mode.label} mode · ${mode.blurb}`}
          >
            {mode.label}
          </span>
        </>
      )}

      {llm && (
        <>
          <Sep />
          <span
            className="font-mono text-[10px]"
            title={`LLM: ${llm}`}
          >
            {llm.split("/").slice(-1)[0]}
          </span>
        </>
      )}

      {running && onCancel && (
        <>
          <Sep />
          <button
            type="button"
            onClick={onCancel}
            className="rounded-full border border-[var(--accent-fg)] px-2.5 py-0.5 text-[10px] font-semibold text-[var(--accent-fg)] transition hover:bg-[var(--accent-fg)] hover:text-[var(--text-body)]"
            aria-label="Stop generation"
            title="Stop generation (Esc)"
          >
            ◼ Stop
          </button>
        </>
      )}
    </div>
  );
}

function Sep() {
  return (
    <span aria-hidden className="text-[var(--cnrs-grey)]">
      ·
    </span>
  );
}

