"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { PageHeader } from "@/components/PageHeader";

export default function SurveyIndexPage() {
  const router = useRouter();
  const [sessionId, setSessionId] = useState("");

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const id = sessionId.trim();
    if (!id) return;
    router.push(`/survey/${encodeURIComponent(id)}`);
  };

  return (
    <main className="relative flex flex-1 flex-col overflow-hidden">
      <PageHeader
        eyebrow="Literature survey"
        title="Survey sessions"
        subtitle="Sessions appear here once a chat in literature_survey mode runs."
      />

      <section className="relative mx-auto flex w-full max-w-3xl flex-1 flex-col items-center justify-center gap-6 px-4 py-12 text-center md:px-6">
        <span
          className="grid h-16 w-16 place-items-center rounded-full"
          style={{ background: "var(--cnrs-yellow)" }}
          aria-hidden
        />

        <div className="flex flex-col items-center gap-2">
          <h2 className="text-xl font-semibold text-[var(--text-body)]">
            No survey session loaded
          </h2>
          <p className="max-w-md text-sm text-[var(--text-muted)]">
            Start a chat in <em className="not-italic">Literature survey</em>{" "}
            mode to create a session. You can also open an existing session by
            entering its identifier below.
          </p>
        </div>

        <form
          onSubmit={submit}
          className="flex w-full max-w-md items-center gap-2 rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] p-2 shadow-[var(--shadow-card)] focus-within:border-[var(--cnrs-blue)]"
        >
          <input
            type="text"
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
            placeholder="Enter session ID"
            className="flex-1 bg-transparent px-3 py-2 text-[15px] leading-relaxed outline-none placeholder:text-[var(--text-muted)]"
          />
          <button
            type="submit"
            disabled={!sessionId.trim()}
            className="rounded-[var(--radius-md)] bg-[var(--cnrs-blue)] px-4 py-2 text-sm font-semibold text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Open →
          </button>
        </form>
      </section>
    </main>
  );
}
