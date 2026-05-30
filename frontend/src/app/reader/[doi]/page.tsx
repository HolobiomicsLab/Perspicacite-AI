"use client";

import { use, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { PageHeader } from "@/components/PageHeader";
import { Markdown } from "@/components/Markdown";
import { papers, type PaperDetail } from "@/lib/api";

type ReaderTab = "abstract" | "full" | "chunks" | "refs" | "figures" | "pdf";

export default function PaperReaderPage({
  params,
}: {
  params: Promise<{ doi: string }>;
}) {
  const { doi: rawDoi } = use(params);
  const doi = decodeURIComponent(rawDoi);

  const [paper, setPaper] = useState<PaperDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<ReaderTab>("abstract");
  const [activeSection, setActiveSection] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- fetch-start flag before async load
    setLoading(true);
    papers
      .byDoi(doi)
      .then((p) => {
        if (!cancelled) {
          setPaper(p);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load");
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [doi]);

  const tocSections = useMemo(() => extractSections(paper?.full_text), [paper?.full_text]);

  const tabs: { id: ReaderTab; label: string; available: boolean }[] = useMemo(
    () => [
      { id: "abstract", label: "Abstract", available: !!paper?.abstract },
      { id: "full", label: "Full text", available: !!paper?.full_text },
      { id: "chunks", label: `Chunks${paper?.chunks ? ` · ${paper.chunks.length}` : ""}`, available: !!paper?.chunks?.length },
      { id: "refs", label: `References${paper?.references ? ` · ${paper.references.length}` : ""}`, available: !!paper?.references?.length },
      { id: "figures", label: `Figures${paper?.capsule?.figures ? ` · ${paper.capsule.figures.length}` : ""}`, available: !!paper?.capsule?.figures?.length },
      { id: "pdf", label: "PDF", available: !!(paper?.pdf_url ?? paper?.oa_url) },
    ],
    [paper],
  );

  // Default tab = first available.
  useEffect(() => {
    if (!paper) return;
    const first = tabs.find((t) => t.available);
    // eslint-disable-next-line react-hooks/set-state-in-effect -- fall back to first available tab when current becomes unavailable
    if (first && !tabs.find((t) => t.id === tab)?.available) setTab(first.id);
  }, [paper, tab, tabs]);

  if (loading) {
    return (
      <main className="flex flex-1 items-center justify-center text-sm text-[var(--text-muted)]">
        Loading paper…
      </main>
    );
  }

  if (error || !paper) {
    return (
      <main className="relative flex flex-1 flex-col">
        <div className="cnrs-halo cnrs-halo--hero opacity-30" aria-hidden />
        <PageHeader eyebrow="Paper" title="Not found" />
        <section className="mx-auto w-full max-w-3xl px-6 py-8">
          <p className="text-sm text-[var(--text-muted)]">
            {error ?? `Could not load DOI ${doi}`}
          </p>
        </section>
      </main>
    );
  }

  return (
    <main className="relative flex flex-1 flex-col">
      <PageHeader
        eyebrow="Reader"
        title={paper.title ?? "(untitled)"}
        subtitle={[
          paper.authors?.slice(0, 4).join(", ") +
            (paper.authors && paper.authors.length > 4
              ? ` +${paper.authors.length - 4}`
              : ""),
          paper.year ? String(paper.year) : "",
          paper.journal ?? "",
          paper.doi ? paper.doi : "",
          typeof paper.citation_count === "number"
            ? `${paper.citation_count.toLocaleString()} cites`
            : "",
        ]
          .filter(Boolean)
          .join(" · ")}
        actions={
          <>
            {paper.doi && (
              <a
                href={`https://doi.org/${paper.doi}`}
                target="_blank"
                rel="noreferrer noopener"
                className="rounded-[var(--radius-md)] border border-[var(--border)] px-4 py-2 text-sm font-medium text-[var(--cnrs-blue)] hover:bg-[var(--cnrs-grey-light)]"
              >
                Open DOI
              </a>
            )}
            {paper.pdf_url || paper.oa_url ? (
              <a
                href={paper.pdf_url ?? paper.oa_url ?? "#"}
                target="_blank"
                rel="noreferrer noopener"
                className="rounded-[var(--radius-md)] bg-[var(--cnrs-blue)] px-4 py-2 text-sm font-semibold text-white hover:bg-[#003a6a]"
              >
                Download PDF
              </a>
            ) : null}
          </>
        }
      />

      <section className="mx-auto grid w-full max-w-6xl flex-1 grid-cols-1 gap-6 px-6 py-6 lg:grid-cols-[260px_1fr]">
        {/* Side TOC + tabs */}
        <aside className="flex flex-col gap-4 lg:sticky lg:top-6 lg:max-h-[calc(100vh-7rem)] lg:overflow-y-auto">
          <nav
            role="tablist"
            aria-label="Reader sections"
            className="flex flex-col gap-1"
          >
            {tabs.map((t) => (
              <button
                key={t.id}
                role="tab"
                aria-selected={tab === t.id}
                disabled={!t.available}
                onClick={() => setTab(t.id)}
                className={[
                  "rounded-[var(--radius-md)] px-3 py-2 text-left text-sm transition",
                  tab === t.id
                    ? "bg-[var(--cnrs-yellow)] font-medium text-[var(--cnrs-blue)]"
                    : t.available
                      ? "text-[var(--text-body)] hover:bg-[var(--cnrs-grey-light)]"
                      : "cursor-not-allowed text-[var(--text-muted)] opacity-50",
                ].join(" ")}
              >
                {t.label}
              </button>
            ))}
          </nav>

          {tab === "full" && tocSections.length > 0 && (
            <div className="border-t border-[var(--border)] pt-3">
              <p className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                Outline
              </p>
              <ol className="flex flex-col gap-px text-xs">
                {tocSections.map((s) => (
                  <li key={s.id}>
                    <a
                      href={`#${s.id}`}
                      onClick={() => setActiveSection(s.id)}
                      className={[
                        "block truncate rounded px-2 py-1 transition",
                        activeSection === s.id
                          ? "bg-[var(--cnrs-yellow)]/40 text-[var(--cnrs-blue)]"
                          : "text-[var(--text-body)] hover:bg-[var(--cnrs-grey-light)]",
                      ].join(" ")}
                    >
                      {s.label}
                    </a>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </aside>

        {/* Main reader pane */}
        <article className="min-w-0">
          {tab === "abstract" && (
            <ReaderCard>
              <h2 className="mb-2 font-mono text-[11px] uppercase tracking-wider text-[var(--text-muted)]">
                Abstract
              </h2>
              {paper.abstract ? (
                <Markdown>{paper.abstract}</Markdown>
              ) : (
                <p className="text-sm text-[var(--text-muted)]">No abstract available.</p>
              )}
            </ReaderCard>
          )}

          {tab === "full" && (
            <ReaderCard>
              {paper.full_text ? (
                <Markdown>{paper.full_text}</Markdown>
              ) : (
                <p className="text-sm text-[var(--text-muted)]">
                  Full text not cached. Try downloading the PDF.
                </p>
              )}
            </ReaderCard>
          )}

          {tab === "chunks" && (
            <ReaderCard>
              <h2 className="mb-3 font-mono text-[11px] uppercase tracking-wider text-[var(--text-muted)]">
                KB chunks · how the retriever sees this paper
              </h2>
              <ol className="flex flex-col gap-3">
                {(paper.chunks ?? []).map((c, i) => (
                  <li
                    key={i}
                    className="rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--bg-soft)] p-3"
                  >
                    <p className="mb-1 flex items-baseline gap-2 text-[11px] uppercase tracking-wider text-[var(--text-muted)]">
                      <span className="rounded-full bg-[var(--cnrs-yellow)] px-1.5 py-px font-mono text-[10px] text-[var(--cnrs-blue)]">
                        #{c.chunk_index ?? i + 1}
                      </span>
                      {c.section && <span>{c.section}</span>}
                    </p>
                    <p className="whitespace-pre-wrap text-[13px] leading-relaxed text-[var(--text-body)]">
                      {c.text}
                    </p>
                  </li>
                ))}
              </ol>
            </ReaderCard>
          )}

          {tab === "refs" && (
            <ReaderCard>
              <h2 className="mb-3 font-mono text-[11px] uppercase tracking-wider text-[var(--text-muted)]">
                References · {paper.references?.length ?? 0}
              </h2>
              <ol className="flex flex-col gap-2">
                {(paper.references ?? []).map((r, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-2 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
                  >
                    <span
                      aria-hidden
                      className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full bg-[var(--cnrs-yellow)] font-mono text-[10px] font-semibold text-[var(--cnrs-blue)]"
                    >
                      {i + 1}
                    </span>
                    <span className="min-w-0 flex-1">
                      {r.doi ? (
                        <Link
                          href={`/reader/${encodeURIComponent(r.doi)}`}
                          className="font-medium text-[var(--cnrs-blue)] hover:underline"
                        >
                          {r.title ?? r.doi}
                        </Link>
                      ) : (
                        <span className="font-medium text-[var(--text-body)]">
                          {r.title ?? "(untitled)"}
                        </span>
                      )}
                      <span className="block text-[11px] text-[var(--text-muted)]">
                        {[r.authors?.[0], r.year].filter(Boolean).join(" · ")}
                      </span>
                    </span>
                  </li>
                ))}
              </ol>
            </ReaderCard>
          )}

          {tab === "figures" && paper.capsule?.figures && (
            <ReaderCard>
              <h2 className="mb-3 font-mono text-[11px] uppercase tracking-wider text-[var(--text-muted)]">
                Figures
              </h2>
              <div className="grid grid-cols-2 gap-3">
                {paper.capsule.figures.map((f) => (
                  <figure
                    key={f.id}
                    className="overflow-hidden rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--bg-soft)]"
                  >
                    {f.url ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={f.url}
                        alt={f.caption ?? f.id}
                        className="w-full"
                      />
                    ) : (
                      <div className="grid h-32 place-items-center text-xs text-[var(--text-muted)]">
                        no preview
                      </div>
                    )}
                    {f.caption && (
                      <figcaption className="border-t border-[var(--border)] px-3 py-2 text-[11px] text-[var(--text-muted)]">
                        {f.caption}
                      </figcaption>
                    )}
                  </figure>
                ))}
              </div>
            </ReaderCard>
          )}

          {tab === "pdf" && (paper.pdf_url || paper.oa_url) && (
            <div className="overflow-hidden rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] shadow-[var(--shadow-card)]">
              <iframe
                title={`PDF · ${paper.title}`}
                src={paper.pdf_url ?? paper.oa_url ?? undefined}
                className="h-[calc(100vh-10rem)] w-full"
              />
            </div>
          )}
        </article>
      </section>
    </main>
  );
}

function ReaderCard({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[var(--shadow-card)]">
      {children}
    </div>
  );
}

// Extract section headings (Markdown-style `## Section`) from full_text
// so the reader can show an inline outline.
function extractSections(text?: string): { id: string; label: string }[] {
  if (!text) return [];
  const headings: { id: string; label: string }[] = [];
  const lines = text.split("\n");
  for (const line of lines) {
    const m = line.match(/^(#{1,3})\s+(.+?)\s*$/);
    if (m) {
      const label = m[2].trim();
      const id = label
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/(^-|-$)/g, "");
      if (id) headings.push({ id, label });
    }
  }
  return headings.slice(0, 50);
}
