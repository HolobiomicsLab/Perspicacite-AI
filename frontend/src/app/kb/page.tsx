"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { PageHeader } from "@/components/PageHeader";
import { kb, type KBSummary } from "@/lib/api";
import { loadPreferences } from "@/lib/preferences";

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

function normalizeListResponse(
  data: KBSummary[] | { kbs: KBSummary[] },
): KBSummary[] {
  return Array.isArray(data) ? data : data.kbs ?? [];
}

export default function KBListPage() {
  const [items, setItems] = useState<KBSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [activeKbName, setActiveKbName] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await kb.list();
      setItems(normalizeListResponse(data));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    setActiveKbName(loadPreferences().defaultKbName);
  }, []);

  return (
    <main className="relative flex flex-1 flex-col">
      <PageHeader
        eyebrow="Knowledge bases"
        title="Your literature corpora."
        subtitle="Curate DOIs and BibTeX into searchable, embedded knowledge bases."
        actions={
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            className="rounded-[var(--radius-md)] bg-[var(--cnrs-blue)] px-4 py-2 text-sm font-semibold text-white shadow-[var(--shadow-card)] transition hover:bg-[var(--cnrs-blue)]/90"
          >
            + New KB
          </button>
        }
      />

      <section className="mx-auto w-full max-w-6xl flex-1 px-6 py-6">
        {error && (
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-red-200 bg-red-50 px-3 py-1 text-xs text-red-700">
            <span aria-hidden>⚠</span>
            <span>{error}</span>
          </div>
        )}

        {loading && items === null ? (
          <LoadingGrid />
        ) : items && items.length === 0 ? (
          <EmptyState onCreate={() => setDrawerOpen(true)} />
        ) : (
          <ul className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {items?.map((k) => (
              <li key={k.name}>
                <KBCard kb={k} isActive={k.name === activeKbName} />
              </li>
            ))}
          </ul>
        )}
      </section>

      {drawerOpen && (
        <NewKBDrawer
          onClose={() => setDrawerOpen(false)}
          onCreated={async () => {
            setDrawerOpen(false);
            await refresh();
          }}
        />
      )}
    </main>
  );
}

function KBCard({ kb: k, isActive }: { kb: KBSummary; isActive: boolean }) {
  return (
    <Link
      href={`/kb/${encodeURIComponent(k.name)}`}
      className="group block h-full rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[var(--shadow-card)] transition hover:border-[var(--cnrs-blue)]/40 hover:shadow-[var(--shadow-elev)]"
    >
      <div className="flex min-h-6 items-center justify-between gap-2">
        {isActive ? (
          <span className="inline-flex items-center gap-1 rounded-full bg-[var(--cnrs-yellow)] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-[var(--cnrs-blue)]">
            <span
              className="h-1.5 w-1.5 rounded-full bg-[var(--cnrs-blue)]"
              aria-hidden
            />
            Active
          </span>
        ) : (
          <span />
        )}
        <span
          className="grid h-2.5 w-2.5 shrink-0 place-items-center rounded-full transition group-hover:scale-125"
          style={{ background: "var(--cnrs-yellow)" }}
          aria-hidden
        />
      </div>
      <h3 className="mt-2 break-words text-xl font-semibold tracking-tight text-[var(--text-body)] group-hover:underline">
        {k.name}
      </h3>

      {k.description ? (
        <p className="mt-2 line-clamp-3 text-sm text-[var(--text-muted)]">
          {k.description}
        </p>
      ) : (
        <p className="mt-2 text-sm italic text-[var(--text-muted)]/70">
          No description.
        </p>
      )}

      <dl className="mt-4 grid grid-cols-2 gap-2 border-t border-[var(--border)] pt-3 text-xs">
        <div>
          <dt className="font-mono uppercase tracking-wider text-[var(--text-muted)]">
            Papers
          </dt>
          <dd className="mt-0.5 text-base font-semibold text-[var(--text-body)]">
            {k.paper_count ?? 0}
          </dd>
        </div>
        <div>
          <dt className="font-mono uppercase tracking-wider text-[var(--text-muted)]">
            Chunks
          </dt>
          <dd className="mt-0.5 text-base font-semibold text-[var(--text-body)]">
            {k.chunk_count ?? 0}
          </dd>
        </div>
      </dl>

      <p className="mt-3 text-[11px] font-mono uppercase tracking-wider text-[var(--text-muted)]">
        Created {formatDate(k.created_at)}
      </p>
    </Link>
  );
}

function LoadingGrid() {
  return (
    <div className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
      <span className="pulse-dot">●</span>
      <span className="pulse-dot" style={{ animationDelay: "0.15s" }}>
        ●
      </span>
      <span className="pulse-dot" style={{ animationDelay: "0.3s" }}>
        ●
      </span>
      <span className="ml-1">Loading knowledge bases…</span>
    </div>
  );
}

function EmptyState({ onCreate }: { onCreate: () => void }) {
  return (
    <div className="flex min-h-[40vh] flex-col items-center justify-center gap-4 rounded-[var(--radius-lg)] border border-dashed border-[var(--border)] bg-[var(--bg-soft)] p-10 text-center">
      <span
        className="grid h-14 w-14 place-items-center rounded-full"
        style={{ background: "var(--cnrs-yellow)" }}
        aria-hidden
      />
      <h2 className="text-xl font-semibold text-[var(--text-body)]">
        No knowledge bases yet
      </h2>
      <p className="max-w-md text-sm text-[var(--text-muted)]">
        Create your first one — group DOIs and BibTeX into a searchable
        corpus, then chat with it from any retrieval mode.
      </p>
      <button
        type="button"
        onClick={onCreate}
        className="rounded-[var(--radius-md)] bg-[var(--cnrs-blue)] px-5 py-2.5 text-sm font-semibold text-white shadow-[var(--shadow-card)] transition hover:bg-[var(--cnrs-blue)]/90"
      >
        + Create knowledge base
      </button>
    </div>
  );
}

function NewKBDrawer({
  onClose,
  onCreated,
}: {
  onClose: () => void;
  onCreated: () => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const trimmed = name.trim();
      if (!trimmed) return;
      setSubmitting(true);
      setError(null);
      try {
        await kb.create({
          name: trimmed,
          description: description.trim() || undefined,
        });
        onCreated();
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to create");
      } finally {
        setSubmitting(false);
      }
    },
    [name, description, onCreated],
  );

  return (
    <div
      className="fixed inset-0 z-30 flex justify-end bg-[var(--cnrs-blue)]/30 backdrop-blur-sm"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="New knowledge base"
    >
      <div
        className="flex h-full w-full max-w-md flex-col border-l border-[var(--border)] bg-[var(--surface)] shadow-[var(--shadow-elev)]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-[var(--border)] px-6 py-4">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-widest text-[var(--text-muted)]">
              Create
            </p>
            <h2 className="mt-0.5 text-lg font-semibold text-[var(--text-body)]">
              New knowledge base
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-[var(--radius-sm)] px-2 py-1 text-sm text-[var(--text-muted)] transition hover:bg-[var(--surface-hover)] hover:text-[var(--text-body)]"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <form onSubmit={submit} className="flex flex-1 flex-col gap-4 p-6">
          <label className="flex flex-col gap-1.5 text-sm">
            <span className="font-medium text-[var(--text-body)]">
              Name <span className="text-red-600">*</span>
            </span>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              autoFocus
              placeholder="e.g. mass-spectrometry-2024"
              className="rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--bg-soft)] px-3 py-2 text-sm outline-none placeholder:text-[var(--text-muted)] focus:border-[var(--cnrs-blue)]"
              disabled={submitting}
            />
          </label>

          <label className="flex flex-col gap-1.5 text-sm">
            <span className="font-medium text-[var(--text-body)]">
              Description
            </span>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={4}
              placeholder="Short description of the corpus scope."
              className="resize-none rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--bg-soft)] px-3 py-2 text-sm outline-none placeholder:text-[var(--text-muted)] focus:border-[var(--cnrs-blue)]"
              disabled={submitting}
            />
          </label>

          {error && (
            <div className="inline-flex items-center gap-2 rounded-full border border-red-200 bg-red-50 px-3 py-1 text-xs text-red-700">
              <span aria-hidden>⚠</span>
              <span>{error}</span>
            </div>
          )}

          <div className="mt-auto flex items-center justify-end gap-2 border-t border-[var(--border)] pt-4">
            <button
              type="button"
              onClick={onClose}
              disabled={submitting}
              className="rounded-[var(--radius-md)] border border-[var(--border)] px-4 py-2 text-sm font-medium text-[var(--text-body)] transition hover:bg-[var(--surface-hover)]"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting || !name.trim()}
              className="inline-flex items-center gap-2 rounded-[var(--radius-md)] bg-[var(--cnrs-blue)] px-5 py-2 text-sm font-semibold text-white transition hover:bg-[var(--cnrs-blue)]/90 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {submitting && (
                <span className="pulse-dot" aria-hidden>
                  ●
                </span>
              )}
              Create
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
