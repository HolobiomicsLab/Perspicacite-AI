import Link from "next/link";

export function PageHeader({
  eyebrow,
  title,
  subtitle,
  actions,
}: {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
}) {
  return (
    <div className="relative border-b border-[var(--border)] bg-[var(--surface)]/60 backdrop-blur">
      <Link
        href="/"
        title="Back to chat"
        aria-label="Back to chat"
        className="absolute right-3 top-2 z-10 p-1 text-[var(--text-muted)] opacity-60 transition hover:text-[var(--text-body)] hover:opacity-100"
      >
        <svg
          viewBox="0 0 24 24"
          width="14"
          height="14"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M18 6 6 18M6 6l12 12" />
        </svg>
      </Link>
      <div className="mx-auto flex max-w-6xl items-end justify-between gap-4 px-6 py-5">
        <div>
          {eyebrow && (
            <p className="font-mono text-[11px] uppercase tracking-widest text-[var(--text-muted)]">
              {eyebrow}
            </p>
          )}
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-[var(--text-body)]">
            {title}
          </h1>
          {subtitle && (
            <p className="mt-1 max-w-2xl text-sm text-[var(--text-muted)]">
              {subtitle}
            </p>
          )}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>
    </div>
  );
}
