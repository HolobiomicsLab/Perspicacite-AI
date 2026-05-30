"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PageHeader } from "@/components/PageHeader";
import {
  conversations as convApi,
  type Conversation,
  type ConvMessage,
} from "@/lib/api";

type ConvDetail = Conversation & { messages: ConvMessage[] };

function relativeTime(iso?: string): string {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const diffSec = Math.round((Date.now() - then) / 1000);
  if (diffSec < 60) return "just now";
  const diffMin = Math.round(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.round(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.round(diffHr / 24);
  if (diffDay < 7) return `${diffDay}d ago`;
  const diffWk = Math.round(diffDay / 7);
  if (diffWk < 5) return `${diffWk}w ago`;
  return new Date(iso).toLocaleDateString();
}

export default function ConversationsPage() {
  const [items, setItems] = useState<Conversation[]>([]);
  const [listLoading, setListLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);

  const [query, setQuery] = useState("");
  const [debounced, setDebounced] = useState("");

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ConvDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const [mobileView, setMobileView] = useState<"list" | "detail">("list");

  // Debounce search input — 250ms, no extra deps.
  useEffect(() => {
    const t = setTimeout(() => setDebounced(query.trim()), 250);
    return () => clearTimeout(t);
  }, [query]);

  const loadList = useCallback(async (q: string) => {
    setListLoading(true);
    setListError(null);
    try {
      if (q) {
        const { results } = await convApi.search(q);
        setItems(results);
      } else {
        const { conversations } = await convApi.list();
        setItems(conversations);
      }
    } catch (err) {
      setListError(err instanceof Error ? err.message : "Failed to load");
      setItems([]);
    } finally {
      setListLoading(false);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- async mount/debounce fetch; setState runs post-await, not synchronously
    void loadList(debounced);
  }, [debounced, loadList]);

  // Fetch selected detail.
  useEffect(() => {
    if (!selectedId) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- clearing derived detail when selection cleared
      setDetail(null);
      setDetailError(null);
      return;
    }
    let cancelled = false;
    setDetailLoading(true);
    setDetailError(null);
    convApi
      .get(selectedId)
      .then((d) => {
        if (!cancelled) setDetail(d);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setDetailError(err instanceof Error ? err.message : "Failed to load");
          setDetail(null);
        }
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  const handleSelect = useCallback((id: string) => {
    setSelectedId(id);
    setMobileView("detail");
  }, []);

  const handleDelete = useCallback(async () => {
    if (!selectedId) return;
    if (!confirm("Delete this conversation? This cannot be undone.")) return;
    try {
      await convApi.remove(selectedId);
      setSelectedId(null);
      setDetail(null);
      setMobileView("list");
      await loadList(debounced);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to delete");
    }
  }, [selectedId, debounced, loadList]);

  const handleClearAll = useCallback(async () => {
    if (
      !confirm(
        "Delete ALL saved conversations? This cannot be undone.",
      )
    )
      return;
    try {
      await convApi.removeAll();
      setSelectedId(null);
      setDetail(null);
      setMobileView("list");
      await loadList(debounced);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to clear");
    }
  }, [debounced, loadList]);

  return (
    <main className="relative flex flex-1 flex-col overflow-hidden">
      <PageHeader
        eyebrow="Saved"
        title="Conversations"
        subtitle="Browse, search and export past chat sessions."
        actions={
          <button
            type="button"
            onClick={handleClearAll}
            disabled={items.length === 0 && !listLoading}
            className="rounded-[var(--radius-md)] border border-[var(--cnrs-blue)] px-4 py-2 text-sm font-medium text-[var(--cnrs-blue)] transition hover:bg-[var(--cnrs-blue)] hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
          >
            Clear all
          </button>
        }
      />

      <section className="mx-auto flex w-full max-w-6xl flex-1 gap-4 overflow-hidden px-4 py-6 md:px-6">
        {/* Left rail */}
        <aside
          className={[
            "flex w-full min-w-0 flex-col gap-3 md:w-[280px] md:shrink-0",
            mobileView === "list" ? "flex" : "hidden md:flex",
          ].join(" ")}
        >
          <div className="rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--surface)] p-2 shadow-[var(--shadow-card)] focus-within:border-[var(--cnrs-blue)]">
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search conversations…"
              className="w-full bg-transparent px-2 py-1.5 text-sm outline-none placeholder:text-[var(--text-muted)]"
            />
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-soft)] p-2 shadow-[var(--shadow-card)]">
            {listLoading ? (
              <ListSkeleton />
            ) : listError ? (
              <p className="px-2 py-3 text-xs text-red-700">⚠ {listError}</p>
            ) : items.length === 0 ? (
              <p className="px-2 py-3 text-xs text-[var(--text-muted)]">
                {debounced
                  ? `No matches for “${debounced}”.`
                  : "No saved conversations yet."}
              </p>
            ) : (
              <ul className="flex flex-col gap-1.5">
                {items.map((c) => (
                  <ConvRow
                    key={c.id}
                    conv={c}
                    active={c.id === selectedId}
                    onSelect={handleSelect}
                  />
                ))}
              </ul>
            )}
          </div>
        </aside>

        {/* Right pane */}
        <div
          className={[
            "min-w-0 flex-1 flex-col",
            mobileView === "detail" ? "flex" : "hidden md:flex",
          ].join(" ")}
        >
          {selectedId ? (
            <DetailPane
              id={selectedId}
              detail={detail}
              loading={detailLoading}
              error={detailError}
              onBack={() => setMobileView("list")}
              onDelete={handleDelete}
            />
          ) : (
            <EmptyState />
          )}
        </div>
      </section>
    </main>
  );
}

function ConvRow({
  conv,
  active,
  onSelect,
}: {
  conv: Conversation;
  active: boolean;
  onSelect: (id: string) => void;
}) {
  return (
    <li>
      <button
        type="button"
        onClick={() => onSelect(conv.id)}
        className={[
          "flex w-full flex-col gap-1.5 rounded-[var(--radius-md)] px-3 py-2.5 text-left transition",
          active
            ? "bg-[var(--cnrs-yellow)] text-[var(--cnrs-blue)]"
            : "text-[var(--text-body)] hover:bg-[var(--cnrs-grey-light)]",
        ].join(" ")}
      >
        <span className="line-clamp-2 text-sm font-medium leading-snug">
          {conv.title?.trim() || "Untitled"}
        </span>
        <span className="flex items-center gap-1.5 text-[11px]">
          {conv.kb_name && (
            <span
              className={[
                "rounded-full px-1.5 py-0.5 font-mono uppercase tracking-wider",
                active
                  ? "bg-[var(--cnrs-blue)] text-white"
                  : "bg-[var(--cnrs-grey-light)] text-[var(--cnrs-blue)]",
              ].join(" ")}
            >
              {conv.kb_name}
            </span>
          )}
          {typeof conv.message_count === "number" && (
            <span
              className={[
                "rounded-full px-1.5 py-0.5 font-mono",
                active
                  ? "bg-[var(--cnrs-blue)]/10 text-[var(--cnrs-blue)]"
                  : "bg-[var(--cnrs-grey-light)] text-[var(--text-muted)]",
              ].join(" ")}
            >
              {conv.message_count} msg
            </span>
          )}
          <span
            className={
              active ? "text-[var(--cnrs-blue)]/70" : "text-[var(--text-muted)]"
            }
          >
            {relativeTime(conv.updated_at ?? conv.created_at)}
          </span>
        </span>
      </button>
    </li>
  );
}

function ListSkeleton() {
  return (
    <ul className="flex flex-col gap-1.5">
      {[0, 1, 2, 3].map((i) => (
        <li
          key={i}
          className="h-14 animate-pulse rounded-[var(--radius-md)] bg-[var(--cnrs-grey-light)]"
        />
      ))}
    </ul>
  );
}

function EmptyState() {
  return (
    <div className="flex h-full min-h-[36vh] flex-1 flex-col items-center justify-center gap-3 rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-soft)] p-6 text-center shadow-[var(--shadow-card)]">
      <span
        className="grid h-14 w-14 place-items-center rounded-full"
        style={{ background: "var(--cnrs-yellow)" }}
        aria-hidden
      />
      <h2 className="text-xl font-semibold text-[var(--cnrs-blue)]">
        Select a conversation
      </h2>
      <p className="max-w-md text-sm text-[var(--text-muted)]">
        Pick a saved session from the left to view, export, or delete it.
      </p>
    </div>
  );
}

function DetailPane({
  id,
  detail,
  loading,
  error,
  onBack,
  onDelete,
}: {
  id: string;
  detail: ConvDetail | null;
  loading: boolean;
  error: string | null;
  onBack: () => void;
  onDelete: () => void;
}) {
  const scrollerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollTo({ top: 0 });
  }, [id]);

  const visibleMessages = useMemo(
    () =>
      (detail?.messages ?? []).filter(
        (m) => m.role === "user" || m.role === "assistant",
      ),
    [detail],
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2 rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2.5 shadow-[var(--shadow-card)]">
        <button
          type="button"
          onClick={onBack}
          className="rounded-[var(--radius-sm)] px-2 py-1 text-xs text-[var(--cnrs-blue)] hover:bg-[var(--cnrs-grey-light)] md:hidden"
        >
          ← Back
        </button>
        <h2 className="min-w-0 flex-1 truncate text-sm font-semibold text-[var(--cnrs-blue)]">
          {detail?.title?.trim() || (loading ? "Loading…" : "Untitled")}
        </h2>
        {detail?.kb_name && (
          <span className="rounded-full bg-[var(--cnrs-grey-light)] px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-[var(--cnrs-blue)]">
            {detail.kb_name}
          </span>
        )}
        <a
          href={convApi.exportUrl(id)}
          className="rounded-[var(--radius-sm)] border border-[var(--border)] px-2.5 py-1 text-xs font-medium text-[var(--cnrs-blue)] transition hover:bg-[var(--cnrs-grey-light)]"
        >
          Export JSON
        </a>
        <button
          type="button"
          onClick={onDelete}
          className="rounded-[var(--radius-sm)] border border-red-200 px-2.5 py-1 text-xs font-medium text-red-700 transition hover:bg-red-50"
        >
          Delete
        </button>
      </div>

      <div
        ref={scrollerRef}
        className="min-h-[40vh] flex-1 overflow-y-auto rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-soft)] p-4 shadow-[var(--shadow-card)] md:p-6"
      >
        {loading ? (
          <p className="flex items-center gap-1 text-xs text-[var(--text-muted)]">
            <span className="pulse-dot">●</span>
            <span className="pulse-dot" style={{ animationDelay: "0.15s" }}>
              ●
            </span>
            <span className="pulse-dot" style={{ animationDelay: "0.3s" }}>
              ●
            </span>
            <span className="ml-1">loading</span>
          </p>
        ) : error ? (
          <p className="text-sm text-red-700">⚠ {error}</p>
        ) : visibleMessages.length === 0 ? (
          <p className="text-sm text-[var(--text-muted)]">
            This conversation has no messages.
          </p>
        ) : (
          <ol className="flex flex-col gap-5">
            {visibleMessages.map((m, i) =>
              m.role === "user" ? (
                <UserBubble key={m.id ?? i} text={m.content} />
              ) : (
                <AssistantBubble key={m.id ?? i} text={m.content} />
              ),
            )}
          </ol>
        )}
      </div>
    </div>
  );
}

function UserBubble({ text }: { text: string }) {
  return (
    <li className="flex justify-end">
      <div className="max-w-[80%] rounded-[var(--radius-lg)] rounded-tr-sm bg-[var(--cnrs-blue)] px-4 py-2.5 text-sm leading-relaxed text-white shadow-[var(--shadow-card)]">
        <p className="whitespace-pre-wrap">{text}</p>
      </div>
    </li>
  );
}

function AssistantBubble({ text }: { text: string }) {
  return (
    <li className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className="rounded-full bg-[var(--cnrs-grey-light)] px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-[var(--cnrs-blue)]">
          assistant
        </span>
      </div>
      <div className="max-w-[88%] rounded-[var(--radius-lg)] rounded-tl-sm border border-[var(--border)] bg-[var(--surface)] px-4 py-3 text-[15px] leading-relaxed text-[var(--text-body)] shadow-[var(--shadow-card)]">
        <p className="whitespace-pre-wrap">{text}</p>
      </div>
    </li>
  );
}
