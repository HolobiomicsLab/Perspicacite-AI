"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  conversations as convApi,
  kb as kbApi,
  type Conversation,
  type KBSummary,
} from "@/lib/api";

type Item =
  | { kind: "chat"; id: string; label: string; sub: string }
  | { kind: "kb"; id: string; label: string; sub: string }
  | { kind: "action"; id: string; label: string; sub: string; href: string }
  | { kind: "go"; id: string; label: string; sub: string; href: string };

const STATIC_ITEMS: Item[] = [
  {
    kind: "action",
    id: "new-chat",
    label: "New chat",
    sub: "Start a fresh conversation",
    href: "/",
  },
  {
    kind: "go",
    id: "go-kb",
    label: "Knowledge bases",
    sub: "Manage corpora",
    href: "/kb",
  },
  {
    kind: "go",
    id: "go-survey",
    label: "Literature survey",
    sub: "Open survey sessions",
    href: "/survey",
  },
  {
    kind: "go",
    id: "go-conv",
    label: "All conversations",
    sub: "Search saved chats",
    href: "/conversations",
  },
];

export function CommandPalette() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const [prevQuery, setPrevQuery] = useState(query);
  if (query !== prevQuery) {
    // Reset highlighted row when the query changes — done during render
    // (React's recommended alternative to a setState-in-effect).
    setPrevQuery(query);
    setActive(0);
  }
  const [chats, setChats] = useState<Conversation[]>([]);
  const [kbs, setKbs] = useState<KBSummary[]>([]);

  // Global ⌘K / Ctrl-K shortcut.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      } else if (e.key === "Escape" && open) {
        e.preventDefault();
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  // Lazy-load chats + KBs on first open.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    (async () => {
      try {
        const [{ conversations }, kbList] = await Promise.all([
          convApi.list(),
          kbApi.list(),
        ]);
        if (cancelled) return;
        setChats(conversations);
        setKbs(kbList);
      } catch {
        /* non-critical */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open]);

  const items: Item[] = useMemo(() => {
    const dynChats: Item[] = chats.slice(0, 50).map((c) => ({
      kind: "chat",
      id: c.id,
      label: c.title?.trim() || "Untitled chat",
      sub: `Resume · ${c.kb_name ?? "default"}`,
    }));
    const dynKbs: Item[] = kbs.slice(0, 50).map((k) => ({
      kind: "kb",
      id: k.name,
      label: k.name,
      sub: k.description ?? "Knowledge base",
    }));
    const all = [...STATIC_ITEMS, ...dynKbs, ...dynChats];
    const q = query.trim().toLowerCase();
    if (!q) return all.slice(0, 60);
    return all
      .filter((i) => (i.label + " " + i.sub).toLowerCase().includes(q))
      .slice(0, 60);
  }, [chats, kbs, query]);

  const choose = useCallback(
    (item: Item) => {
      setOpen(false);
      setQuery("");
      switch (item.kind) {
        case "chat":
          router.push(`/chat/${encodeURIComponent(item.id)}`);
          break;
        case "kb":
          router.push(`/kb/${encodeURIComponent(item.id)}`);
          break;
        case "action":
        case "go":
          router.push(item.href);
          break;
      }
    },
    [router],
  );

  const onListKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((i) => Math.min(items.length - 1, i + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(0, i - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const it = items[active];
      if (it) choose(it);
    }
  };

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
      onClick={() => setOpen(false)}
      className="fixed inset-0 z-50 flex items-start justify-center bg-[rgba(0,40,75,0.45)] p-4 pt-[12vh]"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-lg overflow-hidden rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] shadow-[var(--shadow-elev)]"
      >
        <div className="flex items-center gap-2 border-b border-[var(--border)] px-3 py-2.5">
          <span aria-hidden className="text-[var(--text-muted)]">
            ⌘
          </span>
          <input
            autoFocus
            type="text"
            placeholder="Jump to chat, knowledge base, or action…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onListKey}
            className="flex-1 bg-transparent text-sm outline-none placeholder:text-[var(--text-muted)]"
          />
          <kbd className="rounded border border-[var(--border)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--text-muted)]">
            esc
          </kbd>
        </div>

        <ol className="max-h-[50vh] overflow-y-auto py-1">
          {items.length === 0 && (
            <li className="px-4 py-6 text-center text-xs text-[var(--text-muted)]">
              No matches.
            </li>
          )}
          {items.map((item, idx) => {
            const isActive = idx === active;
            return (
              <li key={`${item.kind}-${item.id}`}>
                <button
                  type="button"
                  onClick={() => choose(item)}
                  onMouseEnter={() => setActive(idx)}
                  className={[
                    "flex w-full items-center justify-between gap-3 px-3 py-2 text-left transition",
                    isActive ? "bg-[var(--cnrs-yellow)]/40" : "",
                  ].join(" ")}
                >
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-[var(--text-body)]">
                      {item.label}
                    </p>
                    <p className="truncate text-[11px] text-[var(--text-muted)]">
                      {item.sub}
                    </p>
                  </div>
                  <KindBadge kind={item.kind} />
                </button>
              </li>
            );
          })}
        </ol>

        <div className="flex items-center justify-between border-t border-[var(--border)] bg-[var(--bg-soft)] px-3 py-1.5 text-[10px] text-[var(--text-muted)]">
          <span>
            <kbd className="rounded border border-[var(--border)] px-1 font-mono">↑↓</kbd>{" "}
            navigate ·{" "}
            <kbd className="rounded border border-[var(--border)] px-1 font-mono">↵</kbd>{" "}
            open
          </span>
          <span>⌘K toggle</span>
        </div>
      </div>
    </div>
  );
}

function KindBadge({ kind }: { kind: Item["kind"] }) {
  const label: Record<Item["kind"], string> = {
    chat: "Chat",
    kb: "KB",
    action: "Action",
    go: "Page",
  };
  return (
    <span className="shrink-0 rounded-full bg-[var(--cnrs-grey-light)] px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-[var(--text-muted)]">
      {label[kind]}
    </span>
  );
}
