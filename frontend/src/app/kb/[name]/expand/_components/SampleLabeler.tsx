"use client";

import type { ExpandSample } from "@/lib/api";

// The ~4 calibration samples spanning the score distribution. Each gets a
// tri-state label (undecided / relevant / not). `labels` is keyed by the
// sample's index in `samples`.
export function SampleLabeler({
  samples,
  labels,
  onLabel,
}: {
  samples: ExpandSample[];
  labels: Map<number, boolean>;
  onLabel: (index: number, relevant: boolean) => void;
}) {
  return (
    <ul className="flex flex-col gap-3">
      {samples.map((s, i) => {
        const label = labels.get(i);
        const abstract =
          s.abstract && s.abstract.length > 280
            ? s.abstract.slice(0, 280) + "…"
            : s.abstract;
        return (
          <li
            key={s.doi ?? i}
            className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] p-4 shadow-[var(--shadow-card)]"
          >
            <div className="flex items-start justify-between gap-3">
              <h3 className="text-[15px] font-semibold leading-snug text-[var(--text-body)]">
                {s.title ?? s.doi ?? "(untitled)"}
              </h3>
              <span className="shrink-0 rounded-full bg-[var(--cnrs-grey-light)] px-2 py-0.5 font-mono text-[11px] text-[var(--cnrs-blue)]">
                {s.score.toFixed(3)}
              </span>
            </div>
            {abstract && (
              <p className="mt-2 text-sm leading-relaxed text-[var(--text-muted)]">
                {abstract}
              </p>
            )}
            <div className="mt-3 flex items-center gap-2">
              <button
                type="button"
                onClick={() => onLabel(i, true)}
                aria-pressed={label === true}
                className={[
                  "rounded-[var(--radius-md)] px-3 py-1.5 text-sm font-medium transition",
                  label === true
                    ? "bg-[var(--cnrs-blue)] text-white"
                    : "border border-[var(--border)] text-[var(--text-body)] hover:bg-[var(--surface-hover)]",
                ].join(" ")}
              >
                Relevant
              </button>
              <button
                type="button"
                onClick={() => onLabel(i, false)}
                aria-pressed={label === false}
                className={[
                  "rounded-[var(--radius-md)] px-3 py-1.5 text-sm font-medium transition",
                  label === false
                    ? "bg-red-600 text-white"
                    : "border border-[var(--border)] text-red-700 hover:bg-red-50",
                ].join(" ")}
              >
                Not relevant
              </button>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
