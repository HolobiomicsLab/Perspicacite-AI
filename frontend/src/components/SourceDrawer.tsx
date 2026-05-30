"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import type { ChatSource } from "@/lib/chat";
import { papers, type PaperDetail } from "@/lib/api";
import { DatabaseGlyph } from "./DatabaseGlyph";
import { JournalFavicon } from "./JournalFavicon";
import { Markdown } from "./Markdown";

// A right-side drawer that opens in place when the user clicks a
// source pill. Lighter than navigating to /reader/[doi]: shows the
// abstract, available chunks, figures and quick links — and a button
// to open the full reader if the user wants more.

type Tab = "info" | "abstract" | "chunks" | "figures";

export function SourceDrawer({
  source,
  onClose,
}: {
  source: ChatSource | null;
  onClose: () => void;
}) {
  const open = source !== null;
  const [tab, setTab] = useState<Tab>("info");
  const [detail, setDetail] = useState<PaperDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reset & fetch on open. We only try the network when there's a DOI
  // since /api/paper requires one; otherwise we show what the chat
  // already gave us.
  useEffect(() => {
    if (!source) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- reset drawer state when source cleared
      setDetail(null);
      setError(null);
      setTab("info");
      return;
    }
    setTab(source.abstract ? "abstract" : "info");
    if (!source.doi) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    papers
      .byDoi(source.doi)
      .then((d) => {
        if (!cancelled) setDetail(d);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message ?? "Failed to load");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [source]);

  // Close on Escape.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!source) return null;

  const abstract = detail?.abstract ?? source.abstract;
  const chunks = detail?.chunks ?? [];
  const figures = detail?.capsule?.figures ?? [];
  const readerHref = source.doi ? `/reader/${encodeURIComponent(source.doi)}` : null;
  const externalHref =
    source.url ??
    source.oa_url ??
    source.pdf_url ??
    (source.doi ? `https://doi.org/${source.doi}` : undefined);

  return (
    <>
      {/* Click-outside scrim, low-opacity so the chat is still legible. */}
      <button
        aria-label="Close source panel"
        onClick={onClose}
        className="fixed inset-0 z-40 bg-black/10 backdrop-blur-[1px]"
      />
      <aside
        role="dialog"
        aria-label="Source detail"
        className="fixed right-0 top-0 z-50 flex h-full w-[min(640px,95vw)] flex-col border-l border-[var(--border)] bg-[var(--surface)] shadow-[var(--shadow-elev)]"
      >
        {/* Header */}
        <header className="border-b border-[var(--border)] px-4 py-3">
          <div className="flex items-start gap-2">
            <div className="min-w-0 flex-1">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                Source
              </p>
              <h3 className="mt-0.5 text-sm font-semibold leading-snug text-[var(--cnrs-blue)]">
                {source.title ?? source.doi ?? "Unknown"}
              </h3>
              <p className="mt-1 text-[11px] text-[var(--text-muted)]">
                {source.authors?.length
                  ? source.authors.length > 1
                    ? `${source.authors[0]} et al.`
                    : source.authors[0]
                  : "Unknown author"}
                {typeof source.year === "number" ? ` · ${source.year}` : ""}
                {typeof source.citation_count === "number"
                  ? ` · ${source.citation_count.toLocaleString()} cites`
                  : ""}
              </p>
              {(detail?.journal ?? source.journal) && (
                <p className="mt-1 inline-flex items-center gap-1 text-[11px] text-[var(--text-body)]">
                  <JournalFavicon name={detail?.journal ?? source.journal ?? ""} size={11} />
                  <span className="italic">{detail?.journal ?? source.journal}</span>
                </p>
              )}
              {/* Provider tags */}
              {(source.discovery_sources?.length ||
                source.providers?.length ||
                source.provider) && (
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {Array.from(
                    new Set(
                      [
                        source.provider,
                        ...(source.providers ?? []),
                        ...(source.discovery_sources ?? []),
                      ].filter((s): s is string => !!s),
                    ),
                  ).map((p) => (
                    <span
                      key={p}
                      className="inline-flex items-center gap-1 rounded-full border border-[var(--border)] bg-[var(--surface)] px-1.5 py-0.5"
                    >
                      <DatabaseGlyph id={p} size={9} />
                      <span className="text-[10px] font-medium">{p}</span>
                    </span>
                  ))}
                </div>
              )}
            </div>
            <button
              type="button"
              onClick={onClose}
              className="rounded-[var(--radius-sm)] p-1 text-[var(--text-muted)] transition hover:bg-[var(--cnrs-grey-light)] hover:text-[var(--cnrs-blue)]"
              aria-label="Close"
            >
              ✕
            </button>
          </div>

          {/* Tabs */}
          <nav className="mt-3 flex gap-1 text-[11px]">
            <TabButton id="info" current={tab} setCurrent={setTab}>
              Info
            </TabButton>
            <TabButton id="abstract" current={tab} setCurrent={setTab} disabled={!abstract}>
              Abstract
            </TabButton>
            <TabButton id="chunks" current={tab} setCurrent={setTab} disabled={chunks.length === 0}>
              Chunks {chunks.length > 0 && `· ${chunks.length}`}
            </TabButton>
            <TabButton id="figures" current={tab} setCurrent={setTab} disabled={figures.length === 0}>
              Figures {figures.length > 0 && `· ${figures.length}`}
            </TabButton>
          </nav>
        </header>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-4 py-3 text-[13px] leading-relaxed text-[var(--text-body)]">
          {loading && !detail && (
            <p className="inline-flex items-center gap-2 text-[var(--text-muted)]">
              <span className="cnrs-sun" aria-hidden />
              Loading paper detail…
            </p>
          )}
          {error && !abstract && (
            <p className="rounded-[var(--radius-md)] border border-red-200 bg-red-50 px-3 py-2 text-red-700">
              {error}
            </p>
          )}

          {tab === "info" && (
            <InfoTab source={source} detail={detail} />
          )}
          {tab === "abstract" && abstract && (
            <Markdown>{abstract}</Markdown>
          )}
          {tab === "chunks" && chunks.length > 0 && (
            <ol className="flex flex-col gap-3">
              {chunks.map((c, i) => (
                <li
                  key={i}
                  className="rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--bg-soft)] px-3 py-2"
                >
                  <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                    {c.section ?? `Chunk ${i + 1}`}
                  </p>
                  <p className="whitespace-pre-wrap">{c.text}</p>
                </li>
              ))}
            </ol>
          )}
          {tab === "figures" && figures.length > 0 && (
            <div className="grid grid-cols-2 gap-2">
              {figures.map((f) => (
                <figure
                  key={f.id}
                  className="overflow-hidden rounded-[var(--radius-md)] border border-[var(--border)]"
                >
                  {f.url && (
                    /* eslint-disable-next-line @next/next/no-img-element */
                    <img
                      src={f.url}
                      alt={f.caption ?? f.id}
                      className="w-full"
                    />
                  )}
                  {f.caption && (
                    <figcaption className="border-t border-[var(--border)] bg-[var(--bg-soft)] px-2 py-1 text-[11px] text-[var(--text-muted)]">
                      {f.caption}
                    </figcaption>
                  )}
                </figure>
              ))}
            </div>
          )}
        </div>

        {/* Footer actions */}
        <footer className="flex flex-wrap items-center gap-2 border-t border-[var(--border)] bg-[var(--bg-soft)] px-4 py-2.5">
          {readerHref && (
            <Link
              href={readerHref}
              className="inline-flex items-center gap-1 rounded-[var(--radius-md)] bg-[var(--cnrs-blue)] px-3 py-1.5 text-xs font-semibold text-white hover:opacity-90"
            >
              Open in reader →
            </Link>
          )}
          {externalHref && (
            <a
              href={externalHref}
              target="_blank"
              rel="noreferrer noopener"
              className="inline-flex items-center gap-1 rounded-[var(--radius-md)] border border-[var(--border)] px-3 py-1.5 text-xs font-medium text-[var(--cnrs-blue)] hover:border-[var(--cnrs-blue)]"
            >
              External ↗
            </a>
          )}
          {typeof source.relevance_score === "number" && (
            <span
              className="ml-auto inline-flex items-center gap-1 rounded-full border border-[var(--cnrs-yellow)] bg-[var(--cnrs-yellow)]/40 px-2 py-0.5 text-[10px] font-mono text-[var(--cnrs-blue)]"
              title="Blended relevance: 60% MiniLM + 25% log citations + 15% BM25"
            >
              ★ {
                source.relevance_score <= 1
                  ? Math.round(source.relevance_score * 100)
                  : source.relevance_score <= 5
                    ? Math.round(source.relevance_score * 20)
                    : Math.round(source.relevance_score)
              }%
            </span>
          )}
        </footer>
      </aside>
    </>
  );
}

function TabButton({
  id,
  current,
  setCurrent,
  disabled,
  children,
}: {
  id: Tab;
  current: Tab;
  setCurrent: (t: Tab) => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  const on = id === current;
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => setCurrent(id)}
      className={[
        "rounded-full px-2.5 py-1 text-[11px] font-medium transition",
        on
          ? "bg-[var(--cnrs-blue)] text-white"
          : "border border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--cnrs-blue)] hover:text-[var(--cnrs-blue)]",
        disabled && "cursor-not-allowed opacity-40",
      ].join(" ")}
    >
      {children}
    </button>
  );
}

function InfoTab({
  source,
  detail,
}: {
  source: ChatSource;
  detail: PaperDetail | null;
}) {
  const rows: Array<[string, React.ReactNode]> = [];
  if (source.doi) rows.push(["DOI", <code key="doi" className="font-mono text-[12px]">{source.doi}</code>]);
  if (detail?.journal ?? source.year)
    rows.push([
      "Venue",
      <span key="v">
        {detail?.journal ?? "—"}
        {typeof source.year === "number" ? ` · ${source.year}` : ""}
      </span>,
    ]);
  if (source.authors?.length)
    rows.push(["Authors", <span key="a">{source.authors.join(", ")}</span>]);
  if (typeof source.citation_count === "number")
    rows.push(["Citations", <span key="c">{source.citation_count.toLocaleString()}</span>]);
  if (source.url) rows.push(["URL", <a key="u" href={source.url} target="_blank" rel="noreferrer noopener" className="text-[var(--cnrs-blue)] underline">link</a>]);
  if (source.pdf_url) rows.push(["PDF", <a key="p" href={source.pdf_url} target="_blank" rel="noreferrer noopener" className="text-[var(--cnrs-blue)] underline">PDF</a>]);
  if (source.oa_url) rows.push(["Open Access", <a key="oa" href={source.oa_url} target="_blank" rel="noreferrer noopener" className="text-[var(--cnrs-blue)] underline">OA copy</a>]);

  if (rows.length === 0) {
    return <p className="text-[var(--text-muted)]">No metadata available.</p>;
  }

  return (
    <dl className="grid grid-cols-[110px_1fr] gap-x-3 gap-y-2 text-[12px]">
      {rows.map(([k, v], i) => (
        <span key={`${k}-${i}`} className="contents">
          <dt className="font-semibold uppercase tracking-wider text-[var(--text-muted)]">
            {k}
          </dt>
          <dd className="min-w-0 break-words text-[var(--text-body)]">{v}</dd>
        </span>
      ))}
    </dl>
  );
}
