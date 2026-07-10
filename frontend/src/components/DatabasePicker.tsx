"use client";

import { useState } from "react";
import {
  DATABASES,
  PRIORITY_DATABASES,
  OTHER_DATABASES,
  type DatabaseDescriptor,
  type DatabaseId,
} from "@/lib/databases";
import { DatabaseGlyph } from "./DatabaseGlyph";

export function DatabasePicker({
  value,
  onChange,
  disabled,
  compact = false,
}: {
  value: DatabaseId[];
  onChange: (next: DatabaseId[]) => void;
  disabled?: boolean;
  compact?: boolean;
}) {
  const selected = new Set(value);
  const [showOthers, setShowOthers] = useState(false);

  const toggle = (id: DatabaseId) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    onChange(Array.from(next));
  };
  const allOn = value.length === DATABASES.length;
  const toggleAll = () =>
    onChange(allOn ? [] : DATABASES.map((d) => d.id));

  // How many of the hidden providers are currently selected — handy
  // signal in the expander toggle so users can see "3 selected" even
  // when the row is collapsed.
  const hiddenSelectedCount = OTHER_DATABASES.filter((d) =>
    selected.has(d.id),
  ).length;

  const gridClass = compact
    ? "grid grid-cols-2 gap-1.5 sm:grid-cols-3"
    : "grid grid-cols-2 gap-1.5 sm:grid-cols-3 lg:grid-cols-3";

  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--surface)] p-3">
      <div className="mb-2 flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          Search databases · {value.length}/{DATABASES.length}
        </p>
        <button
          type="button"
          onClick={toggleAll}
          className="text-[11px] font-medium text-[var(--text-body)] underline-offset-2 hover:underline"
          disabled={disabled}
        >
          {allOn ? "Deselect all" : "Select all"}
        </button>
      </div>

      {/* Priority row — 6 providers in a 2×3 grid, always visible. */}
      <div className={gridClass}>
        {PRIORITY_DATABASES.map((db) => (
          <DatabaseChip
            key={db.id}
            db={db}
            on={selected.has(db.id)}
            disabled={disabled}
            onToggle={() => toggle(db.id)}
          />
        ))}
      </div>

      {/* Expander for the rest. */}
      {OTHER_DATABASES.length > 0 && (
        <div className="mt-2">
          <button
            type="button"
            onClick={() => setShowOthers((s) => !s)}
            className="flex w-full items-center justify-between rounded-[var(--radius-sm)] px-2 py-1.5 text-[11px] font-medium text-[var(--text-muted)] transition hover:bg-[var(--surface-hover)] hover:text-[var(--text-body)]"
            aria-expanded={showOthers}
          >
            <span className="inline-flex items-center gap-1.5">
              <span aria-hidden>{showOthers ? "▾" : "▸"}</span>
              More databases · {OTHER_DATABASES.length}
              {hiddenSelectedCount > 0 && (
                <span className="ml-1 rounded-full bg-[var(--cnrs-yellow)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--cnrs-blue)]">
                  {hiddenSelectedCount} on
                </span>
              )}
            </span>
          </button>
          {showOthers && (
            <div className={`mt-1.5 ${gridClass}`}>
              {OTHER_DATABASES.map((db) => (
                <DatabaseChip
                  key={db.id}
                  db={db}
                  on={selected.has(db.id)}
                  disabled={disabled}
                  onToggle={() => toggle(db.id)}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function DatabaseChip({
  db,
  on,
  disabled,
  onToggle,
}: {
  db: DatabaseDescriptor;
  on: boolean;
  disabled?: boolean;
  onToggle: () => void;
}) {
  return (
    <label
      className={[
        "flex cursor-pointer items-center gap-2 rounded-[var(--radius-sm)] border px-2 py-1.5 text-xs transition",
        on
          ? "border-[var(--cnrs-blue)] bg-[var(--cnrs-yellow)]/30 text-[var(--cnrs-blue)]"
          : "border-[var(--border)] bg-transparent text-[var(--text-body)] hover:bg-[var(--surface-hover)]",
        disabled && "cursor-not-allowed opacity-60",
      ].join(" ")}
      title={db.blurb ? `${db.label} · ${db.blurb}` : db.label}
    >
      <input
        type="checkbox"
        checked={on}
        onChange={onToggle}
        disabled={disabled}
        className="h-3.5 w-3.5 cursor-pointer accent-[var(--cnrs-blue)]"
      />
      <DatabaseGlyph id={db.id} size={11} />
      <span className="min-w-0 flex-1 truncate font-medium">{db.label}</span>
    </label>
  );
}
