"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import React, { useCallback, useEffect, useState } from "react";
import { PageHeader } from "@/components/PageHeader";
import {
  kb,
  type KBChunk,
  type KBPaper,
  type KBStats,
  type KBSummary,
} from "@/lib/api";

type Tab = "overview" | "papers" | "chunks";

function formatDate(iso?: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return iso;
  }
}

function formatBytes(bytes?: number): string {
  if (bytes == null) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const kib = bytes / 1024;
  if (kib < 1024) return `${kib.toFixed(1)} KiB`;
  const mib = kib / 1024;
  if (mib < 1024) return `${mib.toFixed(1)} MiB`;
  return `${(mib / 1024).toFixed(1)} GiB`;
}

export default function KBDetailPage({
  params,
}: {
  params: Promise<{ name: string }>;
}) {
  const { name } = React.use(params);
  const router = useRouter();

  const [summary, setSummary] = useState<KBSummary | null>(null);
  const [stats, setStats] = useState<KBStats | null>(null);
  const [papersData, setPapersData] = useState<KBPaper[]>([]);
  const [chunksData, setChunksData] = useState<KBChunk[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("overview");
  const [chunksLoaded, setChunksLoaded] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [s, st, pp] = await Promise.all([
        kb.get(name),
        kb.stats(name),
        kb.papers(name),
      ]);
      setSummary(s);
      setStats(st);
      setPapersData(pp.papers ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }, [name]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- async mount fetch; setState post-await
    refresh();
  }, [refresh]);

  // Lazy-load chunks the first time the user opens that tab.
  useEffect(() => {
    if (tab !== "chunks" || chunksLoaded) return;
    let cancelled = false;
    (async () => {
      try {
        const data = await kb.chunks(name, 20);
        if (!cancelled) {
          setChunksData(data.chunks ?? []);
          setChunksLoaded(true);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load chunks");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [tab, chunksLoaded, name]);

  const handleDelete = useCallback(async () => {
    if (!confirm(`Delete knowledge base "${name}"? This cannot be undone.`)) {
      return;
    }
    try {
      await kb.remove(name);
      router.push("/kb");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete");
    }
  }, [name, router]);

  return (
    <main className="relative flex flex-1 flex-col">
      <PageHeader
        eyebrow="Knowledge base"
        title={name}
        subtitle={summary?.description ?? undefined}
        actions={
          <>
            <Link
              href={`/kb/${encodeURIComponent(name)}/expand`}
              className="rounded-[var(--radius-md)] border border-[var(--border)] px-4 py-2 text-sm font-medium text-[var(--cnrs-blue)] transition hover:border-[var(--cnrs-blue)] hover:bg-[var(--cnrs-grey-light)]"
            >
              Expand by similarity
            </Link>
            <BuildCapsulesButton name={name} />
            <a
              href={kb.exportUrl(name)}
              className="rounded-[var(--radius-md)] border border-[var(--border)] px-4 py-2 text-sm font-medium text-[var(--cnrs-blue)] transition hover:bg-[var(--cnrs-grey-light)]"
            >
              Export
            </a>
            <ClaimGraphExportButton name={name} />
            <button
              type="button"
              onClick={handleDelete}
              className="rounded-[var(--radius-md)] border border-red-300 px-4 py-2 text-sm font-medium text-red-700 transition hover:bg-red-50"
            >
              Delete
            </button>
          </>
        }
      />

      <section className="mx-auto w-full max-w-6xl flex-1 px-6 py-6">
        {error && (
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-red-200 bg-red-50 px-3 py-1 text-xs text-red-700">
            <span aria-hidden>⚠</span>
            <span>{error}</span>
          </div>
        )}

        <Tabs value={tab} onChange={setTab} />

        <div className="mt-4">
          {tab === "overview" && (
            <OverviewPanel
              summary={summary}
              stats={stats}
              loading={loading}
            />
          )}
          {tab === "papers" && (
            <PapersPanel papers={papersData} loading={loading} />
          )}
          {tab === "chunks" && (
            <ChunksPanel
              chunks={chunksData}
              loading={tab === "chunks" && !chunksLoaded}
            />
          )}
        </div>

        <div className="mt-8">
          <AddPapersCard name={name} onChanged={refresh} />
        </div>
      </section>
    </main>
  );
}

function Tabs({ value, onChange }: { value: Tab; onChange: (t: Tab) => void }) {
  const tabs: { id: Tab; label: string }[] = [
    { id: "overview", label: "Overview" },
    { id: "papers", label: "Papers" },
    { id: "chunks", label: "Chunks" },
  ];
  return (
    <div
      role="tablist"
      aria-label="KB sections"
      className="flex items-center gap-1 border-b border-[var(--border)]"
    >
      {tabs.map((t) => {
        const active = t.id === value;
        return (
          <button
            key={t.id}
            role="tab"
            aria-selected={active}
            type="button"
            onClick={() => onChange(t.id)}
            className={[
              "relative -mb-px px-4 py-2.5 text-sm font-medium transition",
              active
                ? "text-[var(--cnrs-blue)]"
                : "text-[var(--text-muted)] hover:text-[var(--cnrs-blue)]",
            ].join(" ")}
          >
            {t.label}
            {active && (
              <span
                aria-hidden
                className="absolute inset-x-2 -bottom-px h-0.5 rounded-full"
                style={{ background: "var(--cnrs-yellow)" }}
              />
            )}
          </button>
        );
      })}
    </div>
  );
}

function StatTile({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] p-4 shadow-[var(--shadow-card)]">
      <p className="font-mono text-[11px] uppercase tracking-wider text-[var(--text-muted)]">
        {label}
      </p>
      <p className="mt-1 text-2xl font-semibold tracking-tight text-[var(--cnrs-blue)]">
        {value}
      </p>
    </div>
  );
}

function OverviewPanel({
  summary,
  stats,
  loading,
}: {
  summary: KBSummary | null;
  stats: KBStats | null;
  loading: boolean;
}) {
  if (loading && !stats) {
    return <InlineLoader label="Loading overview…" />;
  }
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <StatTile label="Papers" value={stats?.paper_count ?? 0} />
      <StatTile label="Chunks" value={stats?.chunk_count ?? 0} />
      <StatTile
        label="Embedding"
        value={
          <span className="font-mono text-base">
            {stats?.embedding_model ?? summary?.embedding_model ?? "—"}
          </span>
        }
      />
      <StatTile label="Size" value={formatBytes(stats?.total_size_bytes)} />
    </div>
  );
}

function PapersPanel({
  papers,
  loading,
}: {
  papers: KBPaper[];
  loading: boolean;
}) {
  if (loading && papers.length === 0) {
    return <InlineLoader label="Loading papers…" />;
  }
  if (papers.length === 0) {
    return (
      <p className="rounded-[var(--radius-lg)] border border-dashed border-[var(--border)] bg-[var(--bg-soft)] p-6 text-center text-sm text-[var(--text-muted)]">
        No papers yet — add some below.
      </p>
    );
  }
  return (
    <div className="overflow-hidden rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] shadow-[var(--shadow-card)]">
      <table className="w-full text-left text-sm">
        <thead className="border-b border-[var(--border)] bg-[var(--bg-soft)]">
          <tr>
            <th className="px-4 py-2.5 font-mono text-[11px] uppercase tracking-wider text-[var(--text-muted)]">
              Title
            </th>
            <th className="px-4 py-2.5 font-mono text-[11px] uppercase tracking-wider text-[var(--text-muted)]">
              Year
            </th>
            <th className="px-4 py-2.5 font-mono text-[11px] uppercase tracking-wider text-[var(--text-muted)]">
              DOI
            </th>
            <th className="px-4 py-2.5 font-mono text-[11px] uppercase tracking-wider text-[var(--text-muted)]">
              Added
            </th>
          </tr>
        </thead>
        <tbody>
          {papers.map((p) => (
            <tr
              key={p.paper_id}
              className="border-b border-[var(--border)] last:border-0 hover:bg-[var(--bg-soft)]"
            >
              <td className="px-4 py-2.5 align-top text-[var(--cnrs-blue)]">
                {p.title ?? (
                  <span className="italic text-[var(--text-muted)]">
                    (untitled)
                  </span>
                )}
              </td>
              <td className="px-4 py-2.5 align-top text-[var(--text-muted)]">
                {p.year ?? "—"}
              </td>
              <td className="px-4 py-2.5 align-top">
                {p.doi ? (
                  <a
                    href={`https://doi.org/${p.doi}`}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="font-mono text-xs text-[var(--cnrs-violet)] hover:underline"
                  >
                    {p.doi}
                  </a>
                ) : (
                  <span className="text-[var(--text-muted)]">—</span>
                )}
              </td>
              <td className="px-4 py-2.5 align-top text-[var(--text-muted)]">
                {formatDate(p.added_at)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ChunksPanel({
  chunks,
  loading,
}: {
  chunks: KBChunk[];
  loading: boolean;
}) {
  if (loading) {
    return <InlineLoader label="Loading chunks…" />;
  }
  if (chunks.length === 0) {
    return (
      <p className="rounded-[var(--radius-lg)] border border-dashed border-[var(--border)] bg-[var(--bg-soft)] p-6 text-center text-sm text-[var(--text-muted)]">
        No chunks yet.
      </p>
    );
  }
  return (
    <ul className="flex flex-col gap-2">
      {chunks.map((c, i) => {
        const preview = c.text.length > 200 ? c.text.slice(0, 200) + "…" : c.text;
        return (
          <li
            key={c.chunk_id ?? `${c.paper_id}-${c.chunk_index ?? i}`}
            className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] p-4 shadow-[var(--shadow-card)]"
          >
            <div className="mb-1.5 flex items-center gap-2">
              {c.section && (
                <span className="rounded-full bg-[var(--cnrs-grey-light)] px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-[var(--cnrs-blue)]">
                  {c.section}
                </span>
              )}
              <span className="font-mono text-[11px] text-[var(--text-muted)]">
                {c.paper_id}
                {c.chunk_index != null ? ` · #${c.chunk_index}` : ""}
              </span>
            </div>
            <p className="text-sm leading-relaxed text-[var(--text-body)]">
              {preview}
            </p>
          </li>
        );
      })}
    </ul>
  );
}

function InlineLoader({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 px-1 py-4 text-sm text-[var(--text-muted)]">
      <span className="pulse-dot">●</span>
      <span className="pulse-dot" style={{ animationDelay: "0.15s" }}>
        ●
      </span>
      <span className="pulse-dot" style={{ animationDelay: "0.3s" }}>
        ●
      </span>
      <span className="ml-1">{label}</span>
    </div>
  );
}

type SubStatus =
  | { kind: "idle" }
  | { kind: "working" }
  | { kind: "ok"; message: string }
  | { kind: "err"; message: string };

function StatusPill({ status }: { status: SubStatus }) {
  if (status.kind === "idle") return null;
  if (status.kind === "working") {
    return (
      <span className="inline-flex items-center gap-2 rounded-full border border-[var(--border)] bg-[var(--bg-soft)] px-3 py-1 text-xs text-[var(--text-muted)]">
        <span className="pulse-dot" aria-hidden>
          ●
        </span>
        Working…
      </span>
    );
  }
  if (status.kind === "ok") {
    return (
      <span className="inline-flex items-center gap-2 rounded-full border border-[var(--cnrs-green)] bg-[var(--cnrs-green)]/30 px-3 py-1 text-xs text-[var(--cnrs-blue)]">
        ✓ {status.message}
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-red-200 bg-red-50 px-3 py-1 text-xs text-red-700">
      ⚠ {status.message}
    </span>
  );
}

function AddPapersCard({
  name,
  onChanged,
}: {
  name: string;
  onChanged: () => void;
}) {
  const [doiText, setDoiText] = useState("");
  const [doiStatus, setDoiStatus] = useState<SubStatus>({ kind: "idle" });

  const [bibFile, setBibFile] = useState<File | null>(null);
  const [bibStatus, setBibStatus] = useState<SubStatus>({ kind: "idle" });

  const submitDois = useCallback(async () => {
    const dois = doiText
      .split(/\r?\n/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (dois.length === 0) return;
    setDoiStatus({ kind: "working" });
    try {
      const res = await kb.addDois(name, dois);
      setDoiStatus({
        kind: "ok",
        message: `Added ${res.added_papers}${
          res.skipped ? ` (skipped ${res.skipped})` : ""
        }`,
      });
      setDoiText("");
      onChanged();
    } catch (err) {
      setDoiStatus({
        kind: "err",
        message: err instanceof Error ? err.message : "Failed",
      });
    }
  }, [doiText, name, onChanged]);

  const submitBibtex = useCallback(async () => {
    if (!bibFile) return;
    setBibStatus({ kind: "working" });
    try {
      const text = await bibFile.text();
      const res = await kb.addBibtex(name, text);
      setBibStatus({
        kind: "ok",
        message: `Added ${res.added_papers} papers`,
      });
      setBibFile(null);
      onChanged();
    } catch (err) {
      setBibStatus({
        kind: "err",
        message: err instanceof Error ? err.message : "Failed",
      });
    }
  }, [bibFile, name, onChanged]);

  return (
    <div className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[var(--shadow-card)]">
      <div className="mb-4">
        <p className="font-mono text-[11px] uppercase tracking-widest text-[var(--text-muted)]">
          Ingest
        </p>
        <h2 className="mt-0.5 text-lg font-semibold text-[var(--cnrs-blue)]">
          Add papers
        </h2>
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        {/* DOIs */}
        <div className="flex flex-col gap-2">
          <label className="text-sm font-medium text-[var(--cnrs-blue)]">
            DOIs <span className="text-[var(--text-muted)]">(one per line)</span>
          </label>
          <textarea
            value={doiText}
            onChange={(e) => setDoiText(e.target.value)}
            rows={5}
            placeholder={"10.1038/nature12373\n10.1126/science.1234567"}
            className="resize-none rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--bg-soft)] px-3 py-2 font-mono text-xs outline-none placeholder:text-[var(--text-muted)] focus:border-[var(--cnrs-blue)]"
            disabled={doiStatus.kind === "working"}
          />
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={submitDois}
              disabled={doiStatus.kind === "working" || !doiText.trim()}
              className="rounded-[var(--radius-md)] bg-[var(--cnrs-blue)] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[var(--cnrs-blue)]/90 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Add DOIs
            </button>
            <StatusPill status={doiStatus} />
          </div>
        </div>

        {/* BibTeX */}
        <div className="flex flex-col gap-2">
          <label className="text-sm font-medium text-[var(--cnrs-blue)]">
            BibTeX file
          </label>
          <input
            type="file"
            accept=".bib,.bibtex,text/x-bibtex"
            onChange={(e) => setBibFile(e.target.files?.[0] ?? null)}
            disabled={bibStatus.kind === "working"}
            className="block w-full text-sm text-[var(--text-muted)] file:mr-3 file:rounded-[var(--radius-md)] file:border-0 file:bg-[var(--cnrs-grey-light)] file:px-3 file:py-2 file:text-sm file:font-medium file:text-[var(--cnrs-blue)] hover:file:bg-[var(--cnrs-blue-pale)]"
          />
          {bibFile && (
            <p className="font-mono text-xs text-[var(--text-muted)]">
              {bibFile.name} · {formatBytes(bibFile.size)}
            </p>
          )}
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={submitBibtex}
              disabled={bibStatus.kind === "working" || !bibFile}
              className="rounded-[var(--radius-md)] bg-[var(--cnrs-blue)] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[var(--cnrs-blue)]/90 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Add from BibTeX
            </button>
            <StatusPill status={bibStatus} />
          </div>
        </div>

        {/* Local files drop zone */}
        <LocalFilesDropZone name={name} onChanged={onChanged} />
      </div>
    </div>
  );
}

function LocalFilesDropZone({
  name,
  onChanged,
}: {
  name: string;
  onChanged: () => void;
}) {
  const [dragging, setDragging] = useState(false);
  const [status, setStatus] = useState<SubStatus>({ kind: "idle" });

  const upload = useCallback(
    async (files: File[]) => {
      if (!files.length) return;
      setStatus({ kind: "working" });
      try {
        const result = await kb.uploadLocalFiles(name, files);
        setStatus({
          kind: "ok",
          message: result.added_papers
            ? `Added ${result.added_papers} paper(s)`
            : result.job_id
              ? `Job queued: ${result.job_id}`
              : "Uploaded",
        });
        onChanged();
      } catch (err) {
        setStatus({
          kind: "err",
          message: err instanceof Error ? err.message : "Upload failed",
        });
      }
    },
    [name, onChanged],
  );

  return (
    <div className="md:col-span-2 mt-2 flex flex-col gap-2">
      <label className="text-sm font-medium text-[var(--cnrs-blue)]">
        Local files <span className="text-[var(--text-muted)]">(PDF / docx)</span>
      </label>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const files = Array.from(e.dataTransfer.files);
          void upload(files);
        }}
        className={[
          "flex flex-col items-center justify-center gap-1 rounded-[var(--radius-md)] border-2 border-dashed px-6 py-6 text-center transition",
          dragging
            ? "border-[var(--cnrs-blue)] bg-[var(--cnrs-yellow)]/20"
            : "border-[var(--border)] bg-[var(--bg-soft)]",
        ].join(" ")}
      >
        <p className="text-sm text-[var(--text-body)]">
          Drop files here, or{" "}
          <label className="cursor-pointer font-medium text-[var(--cnrs-blue)] underline-offset-2 hover:underline">
            click to choose
            <input
              type="file"
              multiple
              className="hidden"
              onChange={(e) => {
                const files = Array.from(e.target.files ?? []);
                void upload(files);
                e.target.value = "";
              }}
            />
          </label>
        </p>
        <p className="text-[10px] text-[var(--text-muted)]">
          PDFs are parsed and added to this knowledge base.
        </p>
      </div>
      <StatusPill status={status} />
    </div>
  );
}

function BuildCapsulesButton({ name }: { name: string }) {
  const [status, setStatus] = useState<SubStatus>({ kind: "idle" });
  const run = useCallback(async () => {
    setStatus({ kind: "working" });
    try {
      const r = await kb.buildCapsules(name);
      setStatus({
        kind: "ok",
        message: r.job_id ? `Job ${r.job_id.slice(0, 8)}…` : "Started",
      });
    } catch (err) {
      setStatus({
        kind: "err",
        message: err instanceof Error ? err.message : "Failed",
      });
    }
  }, [name]);
  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        onClick={run}
        disabled={status.kind === "working"}
        title="Build per-paper capsules (figures, structured text, provenance)"
        className="rounded-[var(--radius-md)] border border-[var(--border)] px-4 py-2 text-sm font-medium text-[var(--text-body)] transition hover:border-[var(--cnrs-blue)] hover:bg-[var(--cnrs-grey-light)] disabled:cursor-not-allowed disabled:opacity-60"
      >
        Build capsules
      </button>
      {status.kind !== "idle" && <StatusPill status={status} />}
    </div>
  );
}

function ClaimGraphExportButton({ name }: { name: string }) {
  const formats = [
    { value: "nquads", label: "N-Quads (.nq)", desc: "Default — compact quad format" },
    { value: "turtle", label: "Turtle (.ttl)", desc: "Human-readable RDF" },
    { value: "jsonld", label: "JSON-LD (.jsonld)", desc: "Linked Data JSON" },
    { value: "rocrate", label: "RO-Crate (.json)", desc: "Research Object Crate" },
  ] as const;

  return (
    <details className="relative">
      <summary className="cursor-pointer list-none rounded-[var(--radius-md)] border border-[var(--border)] px-4 py-2 text-sm font-medium text-[var(--cnrs-blue)] transition hover:bg-[var(--cnrs-grey-light)]">
        Claim graph ▾
      </summary>
      <div className="absolute right-0 top-full z-20 mt-1 w-52 rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--surface)] shadow-lg">
        {formats.map((f) => (
          <a
            key={f.value}
            href={kb.claimGraphUrl(name, f.value)}
            className="flex flex-col gap-0.5 px-3 py-2 text-sm hover:bg-[var(--cnrs-grey-light)]"
            title={f.desc}
          >
            <span className="font-medium text-[var(--text-body)]">{f.label}</span>
            <span className="text-[11px] text-[var(--text-muted)]">{f.desc}</span>
          </a>
        ))}
      </div>
    </details>
  );
}
