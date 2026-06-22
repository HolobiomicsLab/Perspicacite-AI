"use client";

import type { ExpandCandidate } from "@/lib/api";

// The ranked list of candidates that will be kept at the current cutoff.
export function CandidateList({ candidates }: { candidates: ExpandCandidate[] }) {
  if (candidates.length === 0) {
    return (
      <p className="rounded-[var(--radius-lg)] border border-dashed border-[var(--border)] bg-[var(--bg-soft)] p-6 text-center text-sm text-[var(--text-muted)]">
        No candidates at or above the current cutoff — lower the slider to keep more.
      </p>
    );
  }
  return (
    <ul className="flex flex-col gap-2">
      {candidates.map((c, i) => (
        <li
          key={c.doi ?? i}
          className="flex items-start gap-3 rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--surface)] px-4 py-3"
        >
          <span className="mt-0.5 shrink-0 rounded-full bg-[var(--cnrs-grey-light)] px-2 py-0.5 font-mono text-[11px] text-[var(--cnrs-blue)]">
            {c.score.toFixed(3)}
          </span>
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium leading-snug text-[var(--text-body)]">
              {c.title ?? c.doi ?? "(untitled)"}
            </p>
            {c.doi && (
              <a
                href={`https://doi.org/${c.doi}`}
                target="_blank"
                rel="noreferrer noopener"
                className="mt-0.5 inline-block font-mono text-[11px] text-[var(--cnrs-violet)] hover:underline"
              >
                {c.doi}
              </a>
            )}
          </div>
        </li>
      ))}
    </ul>
  );
}
