"use client";

import { useEffect, useState } from "react";
import { health, type Health } from "@/lib/api";

export function AboutButton() {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        title="About Perspicacité"
        aria-label="About"
        className="grid h-9 w-9 place-items-center rounded-[var(--radius-md)] border border-[var(--border)] bg-transparent text-[var(--text-body)] transition hover:bg-[var(--surface-hover)]"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="16" x2="12" y2="12" />
          <line x1="12" y1="8" x2="12.01" y2="8" />
        </svg>
      </button>
      {open && <AboutModal onClose={() => setOpen(false)} />}
    </>
  );
}

function AboutModal({ onClose }: { onClose: () => void }) {
  const [h, setH] = useState<Health | null>(null);

  useEffect(() => {
    health().then(setH).catch(() => setH(null));
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="About Perspicacité"
      className="fixed inset-0 z-50 flex items-center justify-center bg-[rgba(0,40,75,0.45)] p-4"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-md overflow-hidden rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] shadow-[var(--shadow-elev)]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* halo signature device */}
        <span
          aria-hidden
          className="pointer-events-none absolute -right-16 -top-16 h-48 w-48 rounded-full opacity-60"
          style={{ background: "var(--cnrs-yellow)", filter: "blur(2px)" }}
        />
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          className="absolute right-3 top-3 z-10 grid h-7 w-7 place-items-center rounded-full border border-[var(--border)] bg-[var(--surface)] text-[var(--text-muted)] hover:bg-[var(--surface-hover)] hover:text-[var(--text-body)]"
        >
          ×
        </button>

        <div className="relative p-6">
          <p className="font-mono text-[11px] uppercase tracking-widest text-[var(--text-muted)]">
            About
          </p>
          <h2 className="mt-1 text-2xl font-semibold tracking-tight text-[var(--text-body)]">
            Perspicacité
          </h2>
          <p className="mt-2 text-sm leading-relaxed text-[var(--text-body)]">
            A literature AI assistant that searches your knowledge bases first
            and falls back to the open scholarly web. Six retrieval modes,
            transparent sources, no opaque RAG.
          </p>

          <dl className="mt-5 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-xs">
            <dt className="font-medium text-[var(--text-muted)]">Status</dt>
            <dd className="text-[var(--text-body)]">
              {h ? (
                <span className="inline-flex items-center gap-1.5">
                  <span
                    className="h-1.5 w-1.5 rounded-full"
                    style={{
                      background:
                        h.status === "healthy" ? "#16a34a" : "var(--cnrs-orange)",
                    }}
                    aria-hidden
                  />
                  {h.status}
                </span>
              ) : (
                "—"
              )}
            </dd>
            <dt className="font-medium text-[var(--text-muted)]">Provider</dt>
            <dd className="font-mono text-[var(--text-body)]">
              {h?.llm?.default_provider ?? "—"}
            </dd>
            <dt className="font-medium text-[var(--text-muted)]">Model</dt>
            <dd className="font-mono text-[var(--text-body)]">
              {h?.llm?.default_model ?? "—"}
            </dd>
          </dl>

          <div className="mt-5 flex flex-wrap items-center gap-2">
            <a
              href="https://github.com/holobiomicslab/Perspicacite-AI"
              target="_blank"
              rel="noreferrer noopener"
              className="inline-flex items-center gap-1.5 rounded-[var(--radius-md)] border border-[var(--border)] bg-transparent px-3 py-1.5 text-xs font-medium text-[var(--text-body)] transition hover:bg-[var(--surface-hover)]"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
                <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.1.79-.25.79-.56v-2c-3.2.69-3.88-1.37-3.88-1.37-.52-1.33-1.27-1.69-1.27-1.69-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.02 1.75 2.68 1.24 3.34.95.1-.74.4-1.24.72-1.53-2.55-.29-5.24-1.28-5.24-5.7 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.47.11-3.07 0 0 .97-.31 3.18 1.18.92-.26 1.91-.39 2.89-.39.98 0 1.97.13 2.89.39 2.21-1.49 3.18-1.18 3.18-1.18.63 1.6.23 2.78.11 3.07.74.81 1.19 1.84 1.19 3.1 0 4.43-2.7 5.4-5.27 5.69.41.36.78 1.06.78 2.13v3.16c0 .31.21.66.8.55C20.22 21.39 23.5 17.08 23.5 12 23.5 5.65 18.35.5 12 .5z" />
              </svg>
              <span>GitHub</span>
            </a>
            <a
              href="https://github.com/holobiomicslab/Perspicacite-AI/issues/new"
              target="_blank"
              rel="noreferrer noopener"
              className="inline-flex items-center gap-1.5 rounded-[var(--radius-md)] border border-[var(--border)] bg-transparent px-3 py-1.5 text-xs font-medium text-[var(--text-body)] transition hover:bg-[var(--surface-hover)]"
            >
              Report an issue
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
