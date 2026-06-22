"use client";

import { useState } from "react";
import type { ChatSource } from "@/lib/chat";
import { DatabaseGlyph } from "./DatabaseGlyph";
import { SourceDrawer } from "./SourceDrawer";
import { JournalFavicon } from "./JournalFavicon";

const RELEVANCE_BREAKDOWN =
  "Blended relevance: 60% MiniLM (semantic, query vs title+abstract) + 25% log citation count + 15% BM25 (lexical)";

// Longer abstract preview — about 320 chars, cut on a word boundary
// to leave room for the inline "read more" link.
function abstractPreview(text: string | undefined): string | null {
  if (!text) return null;
  const trimmed = text.trim().replace(/\s+/g, " ");
  if (trimmed.length <= 340) return trimmed;
  const cut = trimmed.slice(0, 320);
  const lastSpace = cut.lastIndexOf(" ");
  return (lastSpace > 240 ? cut.slice(0, lastSpace) : cut) + "…";
}

export function SourcePill({
  source,
  index,
}: {
  source: ChatSource;
  index: number;
}) {
  const [open, setOpen] = useState(false);

  const label =
    source.title ?? source.doi ?? source.paper_id ?? `Source ${index + 1}`;

  const author = source.authors?.length
    ? source.authors.length > 1
      ? `${source.authors[0]} et al.`
      : source.authors[0]
    : undefined;

  // Backend emits two scoring conventions through the same field:
  //   • Retrieval blends (MiniLM + citations + BM25): 0–1 fraction
  //   • Agentic LLM scorer: integer 1–5
  // Map both to a percent so the UI shows a consistent unit.
  const relevancePct =
    typeof source.relevance_score === "number"
      ? source.relevance_score <= 1
        ? Math.round(source.relevance_score * 100)
        : source.relevance_score <= 5
          ? Math.round(source.relevance_score * 20)
          : Math.round(source.relevance_score)
      : null;

  const providers = Array.from(
    new Set(
      [
        source.provider,
        ...(source.providers ?? []),
        ...(source.discovery_sources ?? []),
      ].filter((s): s is string => !!s),
    ),
  );

  const preview = abstractPreview(source.abstract);

  return (
    <>
      <div className="group flex flex-col gap-1 rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2.5 transition hover:border-[var(--cnrs-blue)]/40 hover:shadow-[var(--shadow-card)]">
        {/* Top row: number badge + title (up to 2 lines) + open-drawer arrow */}
        <div className="flex items-start gap-2">
          <span
            aria-hidden
            className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full bg-[var(--cnrs-yellow)] font-mono text-[10px] font-semibold text-[var(--cnrs-blue)]"
          >
            {index + 1}
          </span>
          <button
            type="button"
            onClick={() => setOpen(true)}
            className="line-clamp-2 min-w-0 flex-1 text-left text-[12.5px] font-medium leading-snug text-[var(--text-body)] hover:underline"
            title={label}
          >
            {label}
          </button>
          {/* Top-right arrow now opens the abstract drawer (no longer
              an external link — that one moves to the drawer footer). */}
          <button
            type="button"
            onClick={() => setOpen(true)}
            className="shrink-0 rounded-[var(--radius-sm)] border border-[var(--border)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--text-muted)] hover:border-[var(--cnrs-blue)] hover:text-[var(--text-body)]"
            title="Open abstract in side panel"
            aria-label="Open abstract in side panel"
          >
            <span aria-hidden>↗</span>
          </button>
        </div>

        {/* Single dense meta row: authors · year · cites · journal · DBs · score */}
        <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 pl-7 text-[11px] text-[var(--text-muted)]">
          {author && (
            <span>
              {author}
              {typeof source.year === "number" ? ` · ${source.year}` : ""}
            </span>
          )}
          {typeof source.citation_count === "number" && (
            <span>{source.citation_count.toLocaleString()} cites</span>
          )}
          {source.journal && (
            <span
              className="inline-flex min-w-0 items-center gap-1 italic text-[var(--text-body)]"
              title={source.journal}
            >
              <JournalFavicon name={source.journal} size={10} />
              <span className="truncate">{source.journal}</span>
            </span>
          )}
          {providers.length > 0 && (
            <span
              className="inline-flex items-center"
              title={`Indexed by: ${providers.join(", ")}`}
            >
              {providers.map((p, i) => (
                <span
                  key={p}
                  style={{
                    marginLeft: i === 0 ? 0 : -5,
                    zIndex: providers.length - i,
                  }}
                  className="relative inline-flex"
                >
                  <DatabaseGlyph id={p} size={11} />
                </span>
              ))}
            </span>
          )}
          {relevancePct != null && (
            <span
              className="inline-flex items-center gap-1 rounded-full border border-[var(--cnrs-yellow)] bg-[var(--cnrs-yellow)]/40 px-1.5 py-0 font-mono text-[10px] font-medium text-[var(--cnrs-blue)]"
              title={RELEVANCE_BREAKDOWN}
            >
              <span aria-hidden>★</span>
              <span>relevance {relevancePct}%</span>
            </span>
          )}
        </div>

        {/* Abstract preview — longer now, with an inline text link to
            open the side panel at the end of the snippet (less heavy
            than a full button). */}
        {preview && (
          <p className="mt-1 pl-7 text-[12px] leading-relaxed text-[var(--text-body)]">
            {preview}{" "}
            <button
              type="button"
              onClick={() => setOpen(true)}
              className="font-medium text-[var(--text-body)] underline decoration-[var(--cnrs-yellow)] decoration-2 underline-offset-2 hover:decoration-[var(--cnrs-blue)]"
              title="Open full abstract in side panel"
            >
              read more →
            </button>
          </p>
        )}
        {!preview && (
          <p className="pl-7 text-[11px] text-[var(--text-muted)]">
            <button
              type="button"
              onClick={() => setOpen(true)}
              className="font-medium text-[var(--text-body)] underline decoration-[var(--cnrs-yellow)] decoration-2 underline-offset-2 hover:decoration-[var(--cnrs-blue)]"
            >
              Open detail panel →
            </button>
          </p>
        )}
      </div>

      {open && (
        <SourceDrawer source={source} onClose={() => setOpen(false)} />
      )}
    </>
  );
}
