"use client";

import Link from "next/link";
import React, { useCallback, useMemo, useState } from "react";
import { PageHeader } from "@/components/PageHeader";
import {
  jobs,
  kb,
  type ExpandCandidate,
  type ExpandCommitResult,
  type ExpandDirection,
  type ExpandMethod,
  type ExpandScoreReport,
} from "@/lib/api";
import { CandidateList } from "./_components/CandidateList";
import { SampleLabeler } from "./_components/SampleLabeler";
import { ScoreHistogram } from "./_components/ScoreHistogram";

type Phase = "configure" | "calibrate" | "review" | "result";

const DIRECTIONS: { id: ExpandDirection; label: string }[] = [
  { id: "both", label: "Both" },
  { id: "forward", label: "Forward (cited by)" },
  { id: "backward", label: "Backward (references)" },
];

const METHODS: { id: ExpandMethod; label: string }[] = [
  { id: "hybrid", label: "Hybrid (BM25 + embedding)" },
  { id: "embedding", label: "Embedding" },
  { id: "bm25", label: "BM25" },
];

export default function ExpandSimilarPage({
  params,
}: {
  params: Promise<{ name: string }>;
}) {
  const { name } = React.use(params);

  const [phase, setPhase] = useState<Phase>("configure");
  const [error, setError] = useState<string | null>(null);

  // Phase 1 — configure
  const [direction, setDirection] = useState<ExpandDirection>("both");
  const [maxPerSeed, setMaxPerSeed] = useState(10);
  const [method, setMethod] = useState<ExpandMethod>("hybrid");
  const [scoring, setScoring] = useState(false);

  // Phase 2 — score report + calibration labels
  const [report, setReport] = useState<ExpandScoreReport | null>(null);
  const [labels, setLabels] = useState<Map<number, boolean>>(new Map());

  // Phase 3 — cutoff + commit
  const [cutoff, setCutoff] = useState(0.5);
  const [committing, setCommitting] = useState(false);

  // Phase 4 — result
  const [result, setResult] = useState<ExpandCommitResult | null>(null);

  const runScore = useCallback(async () => {
    setScoring(true);
    setError(null);
    try {
      const { job_id } = await kb.expandSimilarScore(name, {
        direction,
        max_per_seed: maxPerSeed,
        method,
      });
      const rep = (await jobs.waitFor(job_id)) as ExpandScoreReport;
      setReport(rep);
      setLabels(new Map());
      setPhase("calibrate");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Scoring failed");
    } finally {
      setScoring(false);
    }
  }, [name, direction, maxPerSeed, method]);

  const labelSample = useCallback((index: number, relevant: boolean) => {
    setLabels((prev) => {
      const next = new Map(prev);
      next.set(index, relevant);
      return next;
    });
  }, []);

  const placeCutoff = useCallback(async () => {
    if (!report) return;
    setError(null);
    const labeled = Array.from(labels.entries()).map(([i, relevant]) => ({
      score: report.samples[i].score,
      relevant,
    }));
    try {
      const { cutoff: c } = await kb.expandSimilarCutoff(name, labeled);
      setCutoff(c);
      setPhase("review");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not compute cutoff");
    }
  }, [name, report, labels]);

  const kept = useMemo<ExpandCandidate[]>(() => {
    if (!report) return [];
    return report.candidates
      .filter((c) => c.score >= cutoff)
      .sort((a, b) => b.score - a.score);
  }, [report, cutoff]);

  const runCommit = useCallback(async () => {
    if (!report) return;
    setCommitting(true);
    setError(null);
    try {
      const scored = report.candidates.map((c) => ({
        doi: c.doi,
        score: c.score,
      }));
      const { job_id } = await kb.expandSimilarCommit(name, scored, cutoff);
      const res = (await jobs.waitFor(job_id)) as ExpandCommitResult;
      setResult(res);
      setPhase("result");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ingest failed");
    } finally {
      setCommitting(false);
    }
  }, [name, report, cutoff]);

  const restart = useCallback(() => {
    setPhase("configure");
    setReport(null);
    setLabels(new Map());
    setResult(null);
    setError(null);
  }, []);

  return (
    <main className="relative flex flex-1 flex-col overflow-hidden">
      <PageHeader
        eyebrow="Knowledge base"
        title={`Expand "${name}" by similarity`}
        subtitle="Snowball the citation graph, then keep only papers similar to what this KB already contains."
        actions={
          <Link
            href={`/kb/${encodeURIComponent(name)}`}
            className="rounded-[var(--radius-md)] border border-[var(--cnrs-blue)] px-3 py-1.5 text-sm font-medium text-[var(--text-body)] transition hover:bg-[var(--cnrs-blue)] hover:text-white"
          >
            ← Back to KB
          </Link>
        }
      />

      <PhaseStepper phase={phase} />

      <section className="mx-auto w-full max-w-5xl flex-1 px-4 py-6 md:px-6">
        {error && (
          <div className="mb-4 rounded-[var(--radius-md)] border border-red-200 bg-red-50 p-3 text-sm text-red-700">
            ⚠ {error}
          </div>
        )}

        {phase === "configure" && (
          <ConfigurePhase
            direction={direction}
            onDirection={setDirection}
            maxPerSeed={maxPerSeed}
            onMaxPerSeed={setMaxPerSeed}
            method={method}
            onMethod={setMethod}
            scoring={scoring}
            onRun={runScore}
          />
        )}

        {phase === "calibrate" && report && (
          <CalibratePhase
            report={report}
            labels={labels}
            onLabel={labelSample}
            onPlaceCutoff={placeCutoff}
            onBack={() => setPhase("configure")}
          />
        )}

        {phase === "review" && report && (
          <ReviewPhase
            report={report}
            cutoff={cutoff}
            onCutoff={setCutoff}
            kept={kept}
            committing={committing}
            onCommit={runCommit}
            onBack={() => setPhase("calibrate")}
          />
        )}

        {phase === "result" && (
          <ResultPhase
            name={name}
            result={result}
            onRestart={restart}
          />
        )}
      </section>
    </main>
  );
}

// ──────────── Phase stepper ────────────────────────────────────────────

function PhaseStepper({ phase }: { phase: Phase }) {
  const steps: { id: Phase; label: string }[] = [
    { id: "configure", label: "1. Configure" },
    { id: "calibrate", label: "2. Calibrate" },
    { id: "review", label: "3. Review" },
    { id: "result", label: "4. Done" },
  ];
  const activeIdx = steps.findIndex((s) => s.id === phase);
  return (
    <div className="border-b border-[var(--border)] bg-[var(--bg-soft)]">
      <div className="mx-auto flex max-w-5xl items-center gap-2 px-6 py-3 text-xs">
        {steps.map((s, i) => {
          const isActive = i === activeIdx;
          const isDone = i < activeIdx;
          return (
            <div key={s.id} className="flex items-center gap-2">
              <span
                className={[
                  "rounded-full px-2.5 py-1 font-mono text-[11px] uppercase tracking-wider",
                  isActive
                    ? "bg-[var(--cnrs-blue)] text-white"
                    : isDone
                      ? "bg-[var(--cnrs-yellow)] text-[var(--cnrs-blue)]"
                      : "bg-[var(--cnrs-grey-light)] text-[var(--text-muted)]",
                ].join(" ")}
              >
                {s.label}
              </span>
              {i < steps.length - 1 && (
                <span className="text-[var(--text-muted)]" aria-hidden>
                  →
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ──────────── Phase 1 — Configure ──────────────────────────────────────

function ConfigurePhase({
  direction,
  onDirection,
  maxPerSeed,
  onMaxPerSeed,
  method,
  onMethod,
  scoring,
  onRun,
}: {
  direction: ExpandDirection;
  onDirection: (d: ExpandDirection) => void;
  maxPerSeed: number;
  onMaxPerSeed: (n: number) => void;
  method: ExpandMethod;
  onMethod: (m: ExpandMethod) => void;
  scoring: boolean;
  onRun: () => void;
}) {
  return (
    <div className="flex flex-col gap-6">
      <div className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[var(--shadow-card)]">
        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          <label className="flex flex-col gap-2">
            <span className="text-sm font-medium text-[var(--text-body)]">
              Snowball direction
            </span>
            <select
              value={direction}
              onChange={(e) => onDirection(e.target.value as ExpandDirection)}
              disabled={scoring}
              className="rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--bg-soft)] px-3 py-2 text-sm outline-none focus:border-[var(--cnrs-blue)]"
            >
              {DIRECTIONS.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
            </select>
            <span className="text-xs leading-snug text-[var(--text-muted)]">
              Which way to follow citations from this KB&rsquo;s papers — forward
              = papers that cite them, backward = papers they reference, both =
              either.
            </span>
          </label>

          <label className="flex flex-col gap-2">
            <span className="text-sm font-medium text-[var(--text-body)]">
              Max candidates per seed
            </span>
            <input
              type="number"
              min={1}
              max={50}
              value={maxPerSeed}
              onChange={(e) =>
                onMaxPerSeed(
                  Math.max(1, Math.min(50, Number(e.target.value) || 1)),
                )
              }
              disabled={scoring}
              className="rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--bg-soft)] px-3 py-2 text-sm outline-none focus:border-[var(--cnrs-blue)]"
            />
            <span className="text-xs leading-snug text-[var(--text-muted)]">
              How many papers to pull from each existing paper, per direction.
              Higher casts a wider net but is slower and noisier.
            </span>
          </label>

          <label className="flex flex-col gap-2">
            <span className="text-sm font-medium text-[var(--text-body)]">
              Similarity scorer
            </span>
            <select
              value={method}
              onChange={(e) => onMethod(e.target.value as ExpandMethod)}
              disabled={scoring}
              className="rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--bg-soft)] px-3 py-2 text-sm outline-none focus:border-[var(--cnrs-blue)]"
            >
              {METHODS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
            <span className="text-xs leading-snug text-[var(--text-muted)]">
              How candidates are matched to the KB. Hybrid blends keyword (BM25)
              and embedding similarity; embedding is meaning-based; BM25 is
              keyword-based.
            </span>
          </label>
        </div>

        <div className="mt-6 flex items-center gap-3">
          <button
            type="button"
            onClick={onRun}
            disabled={scoring}
            className="rounded-[var(--radius-md)] bg-[var(--cnrs-blue)] px-5 py-2.5 text-sm font-semibold text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {scoring ? (
              <span className="inline-flex items-center gap-2">
                <span className="pulse-dot">●</span>
                <span className="pulse-dot" style={{ animationDelay: "0.15s" }}>
                  ●
                </span>
                <span className="pulse-dot" style={{ animationDelay: "0.3s" }}>
                  ●
                </span>
                <span className="ml-1">Searching &amp; scoring…</span>
              </span>
            ) : (
              "Find candidates"
            )}
          </button>
          {scoring && (
            <span className="text-xs text-[var(--text-muted)]">
              Snowballing the citation graph and scoring against this KB — this
              can take a minute for large KBs.
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// ──────────── Phase 2 — Calibrate ──────────────────────────────────────

function CalibratePhase({
  report,
  labels,
  onLabel,
  onPlaceCutoff,
  onBack,
}: {
  report: ExpandScoreReport;
  labels: Map<number, boolean>;
  onLabel: (index: number, relevant: boolean) => void;
  onPlaceCutoff: () => void;
  onBack: () => void;
}) {
  const total = report.candidates.length;

  if (total === 0) {
    return (
      <div className="flex flex-col gap-4">
        <div className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-soft)] p-6 text-sm text-[var(--text-muted)]">
          {report.seed_count === 0
            ? "This knowledge base has no DOIs to snowball from. Add some papers with DOIs first."
            : "Nothing new to screen — the snowball returned no candidates that aren't already in the KB (or they were all filtered out)."}
        </div>
        <div>
          <button
            type="button"
            onClick={onBack}
            className="rounded-[var(--radius-md)] border border-[var(--cnrs-blue)] px-4 py-2 text-sm font-medium text-[var(--text-body)] transition hover:bg-[var(--cnrs-blue)] hover:text-white"
          >
            ← Adjust settings
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-[var(--text-body)]">
          Score distribution
        </h2>
        <p className="text-xs text-[var(--text-muted)]">
          {total} candidate{total === 1 ? "" : "s"} from {report.seed_count} seed
          {report.seed_count === 1 ? "" : "s"} · scorer: {report.method}
        </p>
      </div>

      <ScoreHistogram buckets={report.histogram} />

      <div>
        <h2 className="mb-1 text-sm font-semibold uppercase tracking-wider text-[var(--text-body)]">
          Calibrate the cutoff
        </h2>
        <p className="mb-3 text-sm text-[var(--text-muted)]">
          Mark each sample below as relevant or not. We place a best-fit cutoff
          from your judgments — you can fine-tune it on the next step.
        </p>
        <SampleLabeler
          samples={report.samples}
          labels={labels}
          onLabel={onLabel}
        />
      </div>

      <div className="flex items-center justify-between gap-3 border-t border-[var(--border)] pt-4">
        <button
          type="button"
          onClick={onBack}
          className="rounded-[var(--radius-md)] border border-[var(--cnrs-blue)] px-4 py-2 text-sm font-medium text-[var(--text-body)] transition hover:bg-[var(--cnrs-blue)] hover:text-white"
        >
          ← Back
        </button>
        <button
          type="button"
          onClick={onPlaceCutoff}
          className="rounded-[var(--radius-md)] bg-[var(--cnrs-blue)] px-5 py-2.5 text-sm font-semibold text-white transition hover:opacity-90"
        >
          Place cutoff →
        </button>
      </div>
    </div>
  );
}

// ──────────── Phase 3 — Review ─────────────────────────────────────────

function ReviewPhase({
  report,
  cutoff,
  onCutoff,
  kept,
  committing,
  onCommit,
  onBack,
}: {
  report: ExpandScoreReport;
  cutoff: number;
  onCutoff: (c: number) => void;
  kept: ExpandCandidate[];
  committing: boolean;
  onCommit: () => void;
  onBack: () => void;
}) {
  const total = report.candidates.length;
  return (
    <div className="flex flex-col gap-6 pb-32">
      <ScoreHistogram buckets={report.histogram} cutoff={cutoff} />

      <div className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] p-4 shadow-[var(--shadow-card)]">
        <div className="flex items-center justify-between gap-3">
          <label
            htmlFor="cutoff-slider"
            className="text-sm font-medium text-[var(--text-body)]"
          >
            Cutoff
          </label>
          <span className="font-mono text-sm text-[var(--text-body)]">
            {cutoff.toFixed(3)}
          </span>
        </div>
        <input
          id="cutoff-slider"
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={cutoff}
          onChange={(e) => onCutoff(parseFloat(e.target.value))}
          className="mt-2 w-full accent-[var(--cnrs-blue)]"
        />
        <p className="mt-2 text-sm text-[var(--text-muted)]">
          Keeping{" "}
          <span className="font-semibold text-[var(--text-body)]">
            {kept.length}
          </span>{" "}
          of {total} candidate{total === 1 ? "" : "s"}.
        </p>
      </div>

      <div>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-[var(--text-body)]">
          Will be ingested
        </h2>
        <CandidateList candidates={kept} />
      </div>

      <div className="sticky bottom-0 left-0 right-0 z-10 -mx-4 border-t border-[var(--border)] bg-[var(--surface)]/95 px-4 py-3 backdrop-blur md:-mx-6 md:px-6">
        <div className="mx-auto flex max-w-5xl items-center justify-between gap-3">
          <button
            type="button"
            onClick={onBack}
            disabled={committing}
            className="rounded-[var(--radius-md)] border border-[var(--cnrs-blue)] px-4 py-2 text-sm font-medium text-[var(--text-body)] transition hover:bg-[var(--cnrs-blue)] hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
          >
            ← Re-label samples
          </button>
          <button
            type="button"
            onClick={onCommit}
            disabled={committing || kept.length === 0}
            className="rounded-[var(--radius-md)] bg-[var(--cnrs-blue)] px-5 py-2.5 text-sm font-semibold text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {committing ? (
              <span className="inline-flex items-center gap-2">
                <span className="pulse-dot">●</span>
                <span className="pulse-dot" style={{ animationDelay: "0.15s" }}>
                  ●
                </span>
                <span className="pulse-dot" style={{ animationDelay: "0.3s" }}>
                  ●
                </span>
                <span className="ml-1">Ingesting…</span>
              </span>
            ) : (
              `Ingest ${kept.length} paper${kept.length === 1 ? "" : "s"}`
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

// ──────────── Phase 4 — Result ─────────────────────────────────────────

function ResultPhase({
  name,
  result,
  onRestart,
}: {
  name: string;
  result: ExpandCommitResult | null;
  onRestart: () => void;
}) {
  const added = result?.added_papers ?? 0;
  const kept = result?.kept ?? 0;
  const failed = result?.failed?.length ?? 0;
  return (
    <div className="flex flex-col gap-6">
      <div className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[var(--shadow-card)]">
        <h2 className="text-lg font-semibold text-[var(--text-body)]">
          Expansion complete
        </h2>
        <div className="mt-4 grid grid-cols-3 gap-3">
          <ResultTile label="Kept" value={kept} />
          <ResultTile label="Added" value={added} />
          <ResultTile label="Failed" value={failed} />
        </div>
        {failed > 0 && (
          <p className="mt-3 text-xs text-[var(--text-muted)]">
            {failed} candidate{failed === 1 ? "" : "s"} could not be fetched and
            were skipped.
          </p>
        )}
      </div>

      <div className="flex items-center gap-3">
        <Link
          href={`/kb/${encodeURIComponent(name)}`}
          className="rounded-[var(--radius-md)] bg-[var(--cnrs-blue)] px-5 py-2.5 text-sm font-semibold text-white transition hover:opacity-90"
        >
          View knowledge base
        </Link>
        <button
          type="button"
          onClick={onRestart}
          className="rounded-[var(--radius-md)] border border-[var(--cnrs-blue)] px-4 py-2 text-sm font-medium text-[var(--text-body)] transition hover:bg-[var(--cnrs-blue)] hover:text-white"
        >
          Expand again
        </button>
      </div>
    </div>
  );
}

function ResultTile({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--bg-soft)] p-4 text-center">
      <p className="font-mono text-[11px] uppercase tracking-wider text-[var(--text-muted)]">
        {label}
      </p>
      <p className="mt-1 text-2xl font-semibold text-[var(--text-body)]">
        {value}
      </p>
    </div>
  );
}
