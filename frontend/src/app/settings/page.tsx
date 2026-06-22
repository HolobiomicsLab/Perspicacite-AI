"use client";

import { useEffect, useState } from "react";
import { PageHeader } from "@/components/PageHeader";
import { DatabasePicker } from "@/components/DatabasePicker";
import {
  loadPreferences,
  savePreferences,
  resetPreferences,
  type Preferences,
  type RelevanceMethod,
  type ScreenMethod,
} from "@/lib/preferences";
import { MODES } from "@/lib/modes";
import { health, type Health, kb as kbApi, type KBSummary } from "@/lib/api";

export default function SettingsPage() {
  const [prefs, setPrefs] = useState<Preferences | null>(null);
  const [h, setH] = useState<Health | null>(null);
  const [kbs, setKbs] = useState<KBSummary[]>([]);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setPrefs(loadPreferences());
    health().then(setH).catch(() => setH(null));
    kbApi.list().then(setKbs).catch(() => setKbs([]));
  }, []);

  const update = <K extends keyof Preferences>(key: K, value: Preferences[K]) => {
    setPrefs((p) => (p ? { ...p, [key]: value } : p));
    setSaved(false);
  };

  const save = () => {
    if (!prefs) return;
    savePreferences(prefs);
    setSaved(true);
    setTimeout(() => setSaved(false), 1500);
  };

  const reset = () => {
    if (confirm("Reset all preferences to defaults?")) {
      setPrefs(resetPreferences());
      setSaved(true);
      setTimeout(() => setSaved(false), 1500);
    }
  };

  if (!prefs) {
    return (
      <main className="flex flex-1 items-center justify-center text-sm text-[var(--text-muted)]">
        Loading preferences…
      </main>
    );
  }

  return (
    <main className="relative flex flex-1 flex-col">
      <PageHeader
        eyebrow="Workspace"
        title="Settings & parameters"
        subtitle="Tune retrieval defaults, display, and integration knobs. Saved to your browser only."
        actions={
          <>
            <button
              type="button"
              onClick={reset}
              className="rounded-[var(--radius-md)] border border-[var(--border)] px-4 py-2 text-sm font-medium text-[var(--text-body)] hover:bg-[var(--surface-hover)]"
            >
              Reset
            </button>
            <button
              type="button"
              onClick={save}
              className="rounded-[var(--radius-md)] bg-[var(--cnrs-blue)] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[#003a6a]"
            >
              {saved ? "Saved ✓" : "Save"}
            </button>
          </>
        }
      />

      <section className="mx-auto w-full max-w-4xl flex-1 space-y-6 px-6 py-8">
        {/* Live LLM info */}
        <Card eyebrow="Live runtime" title="LLM backend">
          <Grid>
            <Field label="Provider">
              <Code>{h?.llm?.default_provider ?? "—"}</Code>
            </Field>
            <Field label="Model">
              <Code>{h?.llm?.default_model ?? "—"}</Code>
            </Field>
            <Field label="Status">
              <span className="inline-flex items-center gap-1.5 text-sm">
                <span
                  aria-hidden
                  className="h-1.5 w-1.5 rounded-full"
                  style={{
                    background:
                      h?.status === "healthy"
                        ? "var(--cnrs-green)"
                        : "var(--cnrs-orange)",
                  }}
                />
                <span>{h?.status ?? "unknown"}</span>
              </span>
            </Field>
          </Grid>
          <p className="mt-3 text-xs text-[var(--text-muted)]">
            Provider / model selection is server-side. To switch, edit{" "}
            <Code>config.yaml</Code> under <Code>llm.providers</Code> and
            restart the backend.
          </p>
        </Card>

        {/* Retrieval defaults */}
        <Card
          eyebrow="Retrieval"
          title="Chat defaults"
          subtitle="Applied to every new conversation. Composer-level toggles override per-request."
        >
          <Grid>
            <Field label="Default mode">
              <Select
                value={prefs.defaultMode}
                onChange={(v) => update("defaultMode", v as Preferences["defaultMode"])}
              >
                {MODES.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))}
              </Select>
            </Field>

            <Field label="Default knowledge base">
              <Select
                value={prefs.defaultKbName ?? ""}
                onChange={(v) => update("defaultKbName", v || null)}
              >
                <option value="">— No KB (web only) —</option>
                <option value="auto">Auto — best match</option>
                {kbs.map((k) => (
                  <option key={k.name} value={k.name}>
                    {k.name}
                  </option>
                ))}
              </Select>
            </Field>

            <Field label={`Max papers · ${prefs.maxPapers}`}>
              <input
                type="range"
                min={1}
                max={25}
                step={1}
                value={prefs.maxPapers}
                onChange={(e) => update("maxPapers", Number(e.target.value))}
                className="w-full accent-[var(--cnrs-blue)]"
              />
            </Field>

            <Field label={`Max papers to download (agentic) · ${prefs.maxPapersToDownload}`}>
              <input
                type="range"
                min={1}
                max={50}
                step={1}
                value={prefs.maxPapersToDownload}
                onChange={(e) =>
                  update("maxPapersToDownload", Number(e.target.value))
                }
                className="w-full accent-[var(--cnrs-blue)]"
              />
            </Field>
          </Grid>

          <div className="mt-4 border-t border-[var(--border)] pt-4">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
              Default databases
            </p>
            <DatabasePicker
              value={prefs.defaultDatabases}
              onChange={(v) => update("defaultDatabases", v)}
              compact
            />
          </div>
        </Card>

        {/* Relevance scoring */}
        <Card
          eyebrow="Relevance"
          title="Scoring & screening"
          subtitle="How the system filters and ranks candidate papers."
        >
          <Grid>
            <Field
              label="Relevance method"
              hint='How to score candidates fetched from databases. Rerank uses MiniLM (cross-encoder); LLM uses one cheap LLM call per paper.'
            >
              <Select
                value={prefs.relevanceMethod}
                onChange={(v) =>
                  update("relevanceMethod", v as RelevanceMethod)
                }
              >
                <option value="bm25">BM25 — lexical, fast, free</option>
                <option value="rerank">Rerank — MiniLM cross-encoder</option>
                <option value="llm">LLM — best, slow, costs tokens</option>
              </Select>
            </Field>

            <Field label={`Min relevance · ${prefs.minRelevance.toFixed(2)}`}>
              <div className="flex h-[38px] items-center">
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={prefs.minRelevance}
                  onChange={(e) => update("minRelevance", Number(e.target.value))}
                  className="w-full accent-[var(--cnrs-blue)]"
                />
              </div>
            </Field>

            <Field
              label="Screen method"
              hint="Applied when an MCP `screen_papers` call runs."
            >
              <Select
                value={prefs.screenMethod}
                onChange={(v) => update("screenMethod", v as ScreenMethod)}
              >
                <option value="bm25">BM25</option>
                <option value="rerank">Rerank (MiniLM)</option>
                <option value="llm">LLM</option>
              </Select>
            </Field>

            <Field label={`Screen threshold · ${prefs.screenThreshold.toFixed(2)}`}>
              <div className="flex h-[38px] items-center">
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={prefs.screenThreshold}
                  onChange={(e) =>
                    update("screenThreshold", Number(e.target.value))
                  }
                  className="w-full accent-[var(--cnrs-blue)]"
                />
              </div>
            </Field>
          </Grid>
        </Card>

        {/* Display */}
        <Card eyebrow="Display" title="Preferences">
          <Grid>
            <Field label="Theme">
              <Select
                value={prefs.theme}
                onChange={(v) => update("theme", v as Preferences["theme"])}
              >
                <option value="system">System</option>
                <option value="light">Light</option>
                <option value="dark">Dark</option>
              </Select>
            </Field>
          </Grid>

          <div className="mt-4 border-t border-[var(--border)] pt-4">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
              Chat behavior
            </p>
            <Grid>
              <Toggle
                label="Show reasoning trail by default"
                checked={prefs.showThinkingByDefault}
                onChange={(v) => update("showThinkingByDefault", v)}
              />
              <Toggle
                label="Estimate tokens while typing"
                checked={prefs.estimateTokensWhileTyping}
                onChange={(v) => update("estimateTokensWhileTyping", v)}
              />
            </Grid>
          </div>
        </Card>

        {/* API keys (read-only — security) */}
        <Card
          eyebrow="Integrations"
          title="API keys & secrets"
          subtitle="For your protection, API keys can only be set in config.yaml or environment variables. The GUI never reads or writes them."
        >
          <ul className="space-y-2 text-sm">
            <KeyRow
              name="OPENROUTER_API_KEY"
              hint="LLM provider — OpenRouter (preferred)"
            />
            <KeyRow
              name="OPENAI_API_KEY"
              hint="LLM provider — OpenAI (fallback)"
            />
            <KeyRow
              name="ANTHROPIC_API_KEY"
              hint="LLM provider — Anthropic"
            />
            <KeyRow
              name="SEMANTIC_SCHOLAR_API_KEY"
              hint="Optional. Lifts rate limits on Semantic Scholar lookups."
            />
            <KeyRow
              name="ELSEVIER_API_KEY"
              hint="Optional. Enables ScienceDirect full-text PDF download."
            />
            <KeyRow
              name="SPRINGER_API_KEY"
              hint="Optional. Enables Springer full-text PDF download."
            />
            <KeyRow
              name="ZOTERO_API_KEY"
              hint="Optional. Required for build_kbs_from_zotero."
            />
          </ul>
          <p className="mt-3 text-xs text-[var(--text-muted)]">
            Edit <Code>~/.perspicacite/config.yaml</Code> or set environment
            variables before <Code>perspicacite serve</Code>. Restart the
            backend after changes.
          </p>
        </Card>
      </section>
    </main>
  );
}

function Card({
  eyebrow,
  title,
  subtitle,
  children,
}: {
  eyebrow: string;
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[var(--shadow-card)]">
      <div className="mb-4">
        <p className="font-mono text-[11px] uppercase tracking-widest text-[var(--text-muted)]">
          {eyebrow}
        </p>
        <h2 className="mt-0.5 text-lg font-semibold text-[var(--text-body)]">
          {title}
        </h2>
        {subtitle && (
          <p className="mt-1 text-xs text-[var(--text-muted)]">{subtitle}</p>
        )}
      </div>
      {children}
    </div>
  );
}

function Grid({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-1 items-start gap-x-6 gap-y-4 sm:grid-cols-2">
      {children}
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="flex items-baseline justify-between gap-2">
        <span className="text-sm font-medium text-[var(--text-body)]">
          {label}
        </span>
      </span>
      {children}
      {hint && (
        <span className="text-[11px] leading-snug text-[var(--text-muted)]">
          {hint}
        </span>
      )}
    </label>
  );
}

function Select({
  value,
  onChange,
  children,
}: {
  value: string;
  onChange: (v: string) => void;
  children: React.ReactNode;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--bg-soft)] px-3 py-2 text-sm text-[var(--text-body)] outline-none focus:border-[var(--cnrs-blue)]"
    >
      {children}
    </select>
  );
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--bg-soft)] px-3 py-2.5">
      <span className="text-sm text-[var(--text-body)]">{label}</span>
      <span
        role="switch"
        aria-checked={checked}
        tabIndex={0}
        onClick={() => onChange(!checked)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") onChange(!checked);
        }}
        className={[
          "relative inline-flex h-5 w-9 cursor-pointer items-center rounded-full transition",
          checked ? "bg-[var(--cnrs-blue)]" : "bg-[var(--cnrs-grey)]",
        ].join(" ")}
      >
        <span
          aria-hidden
          className={[
            "inline-block h-3.5 w-3.5 rounded-full bg-white transition",
            checked ? "translate-x-4" : "translate-x-1",
          ].join(" ")}
        />
      </span>
    </label>
  );
}

function Code({ children }: { children: React.ReactNode }) {
  return (
    <code className="rounded bg-[var(--cnrs-grey-light)] px-1.5 py-0.5 font-mono text-[12px] text-[var(--cnrs-blue)]">
      {children}
    </code>
  );
}

function KeyRow({ name, hint }: { name: string; hint: string }) {
  return (
    <li className="flex items-baseline gap-3">
      <Code>{name}</Code>
      <span className="flex-1 text-xs text-[var(--text-muted)]">{hint}</span>
      <span className="font-mono text-[10px] text-[var(--text-muted)]">
        server-side
      </span>
    </li>
  );
}
