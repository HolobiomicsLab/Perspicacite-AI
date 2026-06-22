"use client";

import { MODES, type RAGMode, accentClasses } from "@/lib/modes";

export function ModeSwitcher({
  value,
  onChange,
  disabled,
}: {
  value: RAGMode;
  onChange: (m: RAGMode) => void;
  disabled?: boolean;
}) {
  return (
    <div
      role="radiogroup"
      aria-label="Retrieval mode"
      className="grid grid-cols-2 gap-2 md:grid-cols-3 lg:grid-cols-6"
    >
      {MODES.map((m) => {
        const selected = value === m.id;
        const a = accentClasses(m.accent);
        return (
          <button
            key={m.id}
            role="radio"
            aria-checked={selected}
            disabled={disabled}
            onClick={() => onChange(m.id)}
            className={[
              "group relative overflow-hidden rounded-[var(--radius-md)] border px-3 py-2.5 text-left transition",
              selected
                ? `${a.bg} ${a.border} ${a.text} shadow-[var(--shadow-card)]`
                : "border-[var(--border)] bg-[var(--surface)] text-[var(--text-body)] hover:border-[var(--cnrs-blue)]/40 hover:bg-[var(--surface-hover)]",
              disabled && "cursor-not-allowed opacity-60",
            ].join(" ")}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-medium tracking-tight">{m.label}</span>
              <span
                className={[
                  "rounded-full px-1.5 py-px text-[10px] font-mono",
                  selected
                    ? "bg-[var(--cnrs-blue)] text-white"
                    : "bg-[var(--cnrs-grey-light)] text-[var(--text-muted)]",
                ].join(" ")}
              >
                {m.latency}
              </span>
            </div>
            <p
              className={[
                "mt-1 text-xs leading-snug",
                selected ? "" : "text-[var(--text-muted)]",
              ].join(" ")}
            >
              {m.blurb}
            </p>
          </button>
        );
      })}
    </div>
  );
}
