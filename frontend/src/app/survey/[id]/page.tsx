"use client";

import { use, useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { PageHeader } from "@/components/PageHeader";
import {
  survey,
  type SurveyPaper,
  type SurveySession,
  type SurveyTheme,
} from "@/lib/api";

type Phase = "review" | "confirm" | "report";

const THEME_COLORS = [
  "var(--cnrs-yellow)",
  "var(--cnrs-blue-pale)",
  "var(--cnrs-green)",
  "var(--cnrs-orange)",
  "var(--cnrs-violet)",
] as const;

function themeColor(index: number): string {
  return THEME_COLORS[index % THEME_COLORS.length];
}

function themeIsViolet(index: number): boolean {
  return THEME_COLORS[index % THEME_COLORS.length] === "var(--cnrs-violet)";
}

export default function SurveyDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const router = useRouter();

  const [session, setSession] = useState<SurveySession | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [phase, setPhase] = useState<Phase>("review");

  const [submittingSelect, setSubmittingSelect] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [report, setReport] = useState<string | null>(null);
  const [generateError, setGenerateError] = useState<string | null>(null);

  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let cancelled = false;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- fetch-start flag before async load
    setLoading(true);
    setError(null);
    survey
      .get(id)
      .then((s) => {
        if (cancelled) return;
        setSession(s);
        // Pre-select recommended papers as a friendly default.
        const recIds = new Set<string>(
          (s.papers ?? [])
            .filter((p) => p.recommended)
            .map((p) => p.id),
        );
        setSelectedIds(recIds);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load session");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  const themeIndex = useMemo(() => {
    const map = new Map<string, number>();
    (session?.themes ?? []).forEach((t, i) => map.set(t.name, i));
    return map;
  }, [session]);

  const selectedPapers = useMemo<SurveyPaper[]>(() => {
    if (!session) return [];
    return session.papers.filter((p) => selectedIds.has(p.id));
  }, [session, selectedIds]);

  const toggle = useCallback((paperId: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(paperId)) next.delete(paperId);
      else next.add(paperId);
      return next;
    });
  }, []);

  const confirmSelection = useCallback(async () => {
    if (selectedIds.size === 0) return;
    setSubmittingSelect(true);
    setError(null);
    try {
      await survey.select(id, Array.from(selectedIds));
      setPhase("confirm");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save selection");
    } finally {
      setSubmittingSelect(false);
    }
  }, [id, selectedIds]);

  const generateReport = useCallback(async () => {
    setGenerating(true);
    setGenerateError(null);
    try {
      const res = await survey.generate(id);
      if (res.error) {
        setGenerateError(res.error);
        return;
      }
      setReport(res.report ?? "");
      setPhase("report");
    } catch (err) {
      setGenerateError(
        err instanceof Error ? err.message : "Failed to generate report",
      );
    } finally {
      setGenerating(false);
    }
  }, [id]);

  const copyReport = useCallback(async () => {
    if (!report) return;
    try {
      await navigator.clipboard.writeText(report);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // no-op — clipboard might be unavailable
    }
  }, [report]);

  // ────────── Render guards ─────────────────────────────────────────────

  if (loading) {
    return (
      <main className="relative flex flex-1 flex-col overflow-hidden">
        <PageHeader
          eyebrow="Literature survey"
          title={`Session ${id}`}
          subtitle="Loading session…"
        />
        <section className="mx-auto w-full max-w-5xl px-4 py-8 md:px-6">
          <div className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-soft)] p-6 text-sm text-[var(--text-muted)]">
            Retrieving survey session…
          </div>
        </section>
      </main>
    );
  }

  if (error || !session) {
    return (
      <main className="relative flex flex-1 flex-col overflow-hidden">
        <PageHeader
          eyebrow="Literature survey"
          title={`Session ${id}`}
          subtitle="Unable to load this survey session."
          actions={
            <button
              type="button"
              onClick={() => router.push("/survey")}
              className="rounded-[var(--radius-md)] border border-[var(--cnrs-blue)] px-3 py-1.5 text-sm font-medium text-[var(--cnrs-blue)] transition hover:bg-[var(--cnrs-blue)] hover:text-white"
            >
              ← Back
            </button>
          }
        />
        <section className="mx-auto w-full max-w-5xl px-4 py-8 md:px-6">
          <div className="rounded-[var(--radius-lg)] border border-red-200 bg-red-50 p-6 text-sm text-red-700">
            ⚠ {error ?? "Session not found"}
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="relative flex flex-1 flex-col overflow-hidden">
      <PageHeader
        eyebrow="Literature survey"
        title={`Session ${id}`}
        subtitle={
          session.query ? `Query: ${session.query}` : "Survey session"
        }
        actions={
          <button
            type="button"
            onClick={() => router.push("/survey")}
            className="rounded-[var(--radius-md)] border border-[var(--cnrs-blue)] px-3 py-1.5 text-sm font-medium text-[var(--cnrs-blue)] transition hover:bg-[var(--cnrs-blue)] hover:text-white"
          >
            Restart
          </button>
        }
      />

      <PhaseStepper phase={phase} />

      {phase === "review" && (
        <ReviewPhase
          session={session}
          selectedIds={selectedIds}
          onToggle={toggle}
          onConfirm={confirmSelection}
          submitting={submittingSelect}
          themeIndex={themeIndex}
        />
      )}

      {phase === "confirm" && (
        <ConfirmPhase
          papers={selectedPapers}
          onBack={() => setPhase("review")}
          onGenerate={generateReport}
          generating={generating}
          error={generateError}
        />
      )}

      {phase === "report" && (
        <ReportPhase
          report={report ?? ""}
          onCopy={copyReport}
          copied={copied}
          onRestart={() => router.push("/survey")}
        />
      )}
    </main>
  );
}

// ──────────── Phase stepper ────────────────────────────────────────────

function PhaseStepper({ phase }: { phase: Phase }) {
  const steps: { id: Phase; label: string }[] = [
    { id: "review", label: "1. Review" },
    { id: "confirm", label: "2. Confirm" },
    { id: "report", label: "3. Report" },
  ];
  const activeIdx = steps.findIndex((s) => s.id === phase);
  return (
    <div className="border-b border-[var(--border)] bg-[var(--bg-soft)]">
      <div className="mx-auto flex max-w-6xl items-center gap-2 px-6 py-3 text-xs">
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

// ──────────── Phase 1 — Review ─────────────────────────────────────────

function ReviewPhase({
  session,
  selectedIds,
  onToggle,
  onConfirm,
  submitting,
  themeIndex,
}: {
  session: SurveySession;
  selectedIds: Set<string>;
  onToggle: (id: string) => void;
  onConfirm: () => void;
  submitting: boolean;
  themeIndex: Map<string, number>;
}) {
  const total = session.papers.length;
  const count = selectedIds.size;

  return (
    <>
      <section className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-6 pb-32 md:px-6">
        {/* Themes */}
        <div>
          <div className="mb-3 flex items-baseline justify-between gap-3">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-[var(--cnrs-blue)]">
              Themes
            </h2>
            <p className="text-xs text-[var(--text-muted)]">
              {session.themes.length} detected
            </p>
          </div>
          {session.themes.length === 0 ? (
            <p className="text-sm text-[var(--text-muted)]">
              No themes were detected for this session.
            </p>
          ) : (
            <div className="flex flex-wrap gap-2">
              {session.themes.map((t, i) => (
                <ThemeChip key={t.name + i} theme={t} index={i} />
              ))}
            </div>
          )}
        </div>

        {/* Papers */}
        <div>
          <div className="mb-3 flex items-baseline justify-between gap-3">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-[var(--cnrs-blue)]">
              Papers
            </h2>
            <p className="text-xs text-[var(--text-muted)]">
              {total} candidate{total === 1 ? "" : "s"}
            </p>
          </div>

          {total === 0 ? (
            <div className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-soft)] p-6 text-sm text-[var(--text-muted)]">
              No papers were returned for this session.
            </div>
          ) : (
            <ul className="flex flex-col gap-3">
              {session.papers.map((p) => (
                <PaperCard
                  key={p.id}
                  paper={p}
                  selected={selectedIds.has(p.id)}
                  onToggle={() => onToggle(p.id)}
                  themeIndex={themeIndex}
                />
              ))}
            </ul>
          )}
        </div>
      </section>

      {/* Sticky action bar */}
      <div className="sticky bottom-0 left-0 right-0 z-10 border-t border-[var(--border)] bg-[var(--surface)]/95 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-6 py-3">
          <p className="text-sm text-[var(--text-body)]">
            <span className="font-semibold text-[var(--cnrs-blue)]">
              {count}
            </span>{" "}
            of {total} selected
          </p>
          <button
            type="button"
            onClick={onConfirm}
            disabled={count === 0 || submitting}
            className="rounded-[var(--radius-md)] bg-[var(--cnrs-blue)] px-5 py-2.5 text-sm font-semibold text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {submitting ? "Saving…" : "Next: confirm selection →"}
          </button>
        </div>
      </div>
    </>
  );
}

function ThemeChip({ theme, index }: { theme: SurveyTheme; index: number }) {
  const bg = themeColor(index);
  const violet = themeIsViolet(index);
  return (
    <span
      className="inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium"
      style={{
        background: bg,
        color: violet ? "white" : "var(--cnrs-blue)",
      }}
      title={theme.description ?? undefined}
    >
      <span className="truncate">{theme.name}</span>
      {typeof theme.paper_count === "number" && (
        <span
          className="rounded-full px-1.5 py-0.5 font-mono text-[10px]"
          style={{
            background: violet ? "rgba(255,255,255,0.2)" : "rgba(0,40,75,0.1)",
          }}
        >
          {theme.paper_count}
        </span>
      )}
    </span>
  );
}

function PaperCard({
  paper,
  selected,
  onToggle,
  themeIndex,
}: {
  paper: SurveyPaper;
  selected: boolean;
  onToggle: () => void;
  themeIndex: Map<string, number>;
}) {
  const authors = (paper.authors ?? []).slice(0, 3).join(", ");
  const extraAuthors = (paper.authors?.length ?? 0) > 3 ? " et al." : "";

  return (
    <li>
      <label
        className={[
          "flex cursor-pointer items-start gap-3 rounded-[var(--radius-lg)] border bg-[var(--surface)] p-4 transition hover:border-[var(--cnrs-blue)]/40 hover:shadow-[var(--shadow-card)]",
          selected
            ? "border-[var(--cnrs-blue)] shadow-[var(--shadow-card)]"
            : "border-[var(--border)]",
        ].join(" ")}
      >
        <input
          type="checkbox"
          checked={selected}
          onChange={onToggle}
          className="mt-1 h-4 w-4 shrink-0 accent-[var(--cnrs-blue)]"
        />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-start justify-between gap-2">
            <h3 className="text-[15px] font-semibold leading-snug text-[var(--cnrs-blue)]">
              {paper.title ?? paper.doi ?? paper.id}
            </h3>
            {paper.recommended && (
              <span
                className="rounded-full bg-[var(--cnrs-yellow)] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-[var(--cnrs-blue)]"
                title={paper.reason ?? "Recommended by the survey agent"}
              >
                Recommended
              </span>
            )}
          </div>

          {(authors || paper.year) && (
            <p className="mt-1 text-xs text-[var(--text-muted)]">
              {authors}
              {extraAuthors}
              {authors && paper.year ? " · " : ""}
              {paper.year ?? ""}
            </p>
          )}

          <div className="mt-2 flex flex-wrap items-center gap-3 text-[11px] text-[var(--text-muted)]">
            {typeof paper.citation_count === "number" && (
              <span className="font-mono">
                {paper.citation_count} citation
                {paper.citation_count === 1 ? "" : "s"}
              </span>
            )}
            {typeof paper.relevance_score === "number" && (
              <span className="font-mono">
                relevance {paper.relevance_score.toFixed(2)}
              </span>
            )}
            {paper.doi && (
              <a
                href={`https://doi.org/${paper.doi}`}
                target="_blank"
                rel="noreferrer noopener"
                onClick={(e) => e.stopPropagation()}
                className="font-mono text-[var(--cnrs-blue)] underline-offset-2 hover:underline"
              >
                {paper.doi}
              </a>
            )}
          </div>

          {paper.themes && paper.themes.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {paper.themes.map((tname, i) => {
                const idx = themeIndex.get(tname) ?? i;
                const violet = themeIsViolet(idx);
                return (
                  <span
                    key={tname + i}
                    className="rounded-full px-2 py-0.5 text-[10px] font-medium"
                    style={{
                      background: themeColor(idx),
                      color: violet ? "white" : "var(--cnrs-blue)",
                    }}
                  >
                    {tname}
                  </span>
                );
              })}
            </div>
          )}
        </div>
      </label>
    </li>
  );
}

// ──────────── Phase 2 — Confirm ────────────────────────────────────────

function ConfirmPhase({
  papers,
  onBack,
  onGenerate,
  generating,
  error,
}: {
  papers: SurveyPaper[];
  onBack: () => void;
  onGenerate: () => void;
  generating: boolean;
  error: string | null;
}) {
  return (
    <section className="mx-auto flex w-full max-w-4xl flex-col gap-6 px-4 py-6 md:px-6">
      <div>
        <h2 className="text-sm font-semibold uppercase tracking-wider text-[var(--cnrs-blue)]">
          Confirm your selection
        </h2>
        <p className="mt-1 text-sm text-[var(--text-muted)]">
          {papers.length} paper{papers.length === 1 ? "" : "s"} will be used to
          generate the deep report.
        </p>
      </div>

      <ul className="flex flex-col gap-2">
        {papers.map((p, i) => (
          <li
            key={p.id}
            className="flex items-start gap-3 rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--surface)] px-4 py-3"
          >
            <span className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full bg-[var(--cnrs-yellow)] font-mono text-[11px] font-semibold text-[var(--cnrs-blue)]">
              {i + 1}
            </span>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium text-[var(--cnrs-blue)]">
                {p.title ?? p.id}
              </p>
              {p.doi && (
                <p className="mt-0.5 font-mono text-[11px] text-[var(--text-muted)]">
                  {p.doi}
                </p>
              )}
            </div>
          </li>
        ))}
      </ul>

      {error && (
        <div className="rounded-[var(--radius-md)] border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          ⚠ {error}
        </div>
      )}

      <div className="flex items-center justify-between gap-3 border-t border-[var(--border)] pt-4">
        <button
          type="button"
          onClick={onBack}
          disabled={generating}
          className="rounded-[var(--radius-md)] border border-[var(--cnrs-blue)] px-4 py-2 text-sm font-medium text-[var(--cnrs-blue)] transition hover:bg-[var(--cnrs-blue)] hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
        >
          ← Back to review
        </button>
        <button
          type="button"
          onClick={onGenerate}
          disabled={generating || papers.length === 0}
          className="rounded-[var(--radius-md)] bg-[var(--cnrs-blue)] px-5 py-2.5 text-sm font-semibold text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {generating ? (
            <span className="inline-flex items-center gap-2">
              <span className="pulse-dot">●</span>
              <span
                className="pulse-dot"
                style={{ animationDelay: "0.15s" }}
              >
                ●
              </span>
              <span className="pulse-dot" style={{ animationDelay: "0.3s" }}>
                ●
              </span>
              <span className="ml-1">Generating…</span>
            </span>
          ) : (
            "Generate deep report"
          )}
        </button>
      </div>
    </section>
  );
}

// ──────────── Phase 3 — Report ─────────────────────────────────────────

function ReportPhase({
  report,
  onCopy,
  copied,
  onRestart,
}: {
  report: string;
  onCopy: () => void;
  copied: boolean;
  onRestart: () => void;
}) {
  return (
    <section className="mx-auto flex w-full max-w-3xl flex-col gap-4 px-4 py-6 md:px-6">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-[var(--cnrs-blue)]">
          Deep report
        </h2>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onCopy}
            className="rounded-[var(--radius-md)] border border-[var(--cnrs-blue)] px-3 py-1.5 text-sm font-medium text-[var(--cnrs-blue)] transition hover:bg-[var(--cnrs-blue)] hover:text-white"
          >
            {copied ? "Copied ✓" : "Copy report"}
          </button>
          <button
            type="button"
            onClick={onRestart}
            className="rounded-[var(--radius-md)] bg-[var(--cnrs-blue)] px-3 py-1.5 text-sm font-semibold text-white transition hover:opacity-90"
          >
            Start new survey
          </button>
        </div>
      </div>

      {report ? (
        <article className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[var(--shadow-card)]">
          <div className="whitespace-pre-wrap text-[15px] leading-relaxed text-[var(--text-body)]">
            {report}
          </div>
        </article>
      ) : (
        <div className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-soft)] p-6 text-sm text-[var(--text-muted)]">
          The report came back empty.
        </div>
      )}
    </section>
  );
}
