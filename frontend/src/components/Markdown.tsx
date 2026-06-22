"use client";

import { Component, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize from "rehype-sanitize";

// Tailored markdown renderer for streaming assistant answers. Tight
// rhythm, CNRS-blue headings, prose-grade links, syntax-friendly
// inline code, GFM tables, ordered/unordered lists, blockquotes.
//
// All HTML is sanitized via rehypeSanitize — no unsafe markup ever
// renders from streamed model output.

const components: Components = {
  h1: ({ children }) => (
    <h1 className="mt-6 mb-3 text-2xl font-semibold tracking-tight text-[var(--accent-fg)]">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="mt-5 mb-2 text-xl font-semibold tracking-tight text-[var(--accent-fg)]">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="mt-4 mb-1.5 text-base font-semibold text-[var(--accent-fg)]">
      {children}
    </h3>
  ),
  h4: ({ children }) => (
    <h4 className="mt-3 mb-1 text-sm font-semibold text-[var(--accent-fg)]">
      {children}
    </h4>
  ),
  p: ({ children }) => (
    <p className="my-3 text-[15px] leading-relaxed text-[var(--text-body)]">
      {children}
    </p>
  ),
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer noopener"
      className="font-medium text-[var(--accent-fg)] underline decoration-[var(--cnrs-yellow)] decoration-2 underline-offset-2 transition hover:decoration-[var(--accent-fg)]"
    >
      {children}
    </a>
  ),
  ul: ({ children }) => (
    <ul className="my-3 ml-5 list-disc space-y-1 text-[15px] leading-relaxed text-[var(--text-body)] marker:text-[var(--cnrs-yellow)]">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="my-3 ml-5 list-decimal space-y-1 text-[15px] leading-relaxed text-[var(--text-body)] marker:text-[var(--cnrs-blue)] marker:font-medium">
      {children}
    </ol>
  ),
  li: ({ children }) => <li className="pl-1">{children}</li>,
  blockquote: ({ children }) => (
    <blockquote className="my-3 border-l-4 border-[var(--cnrs-yellow)] bg-[var(--bg-soft)] px-4 py-2 text-[14px] italic text-[var(--text-body)]">
      {children}
    </blockquote>
  ),
  code: (props) => {
    const { children, className } = props as {
      children?: React.ReactNode;
      className?: string;
    };
    const isBlock = !!className;
    if (isBlock) {
      return (
        <code className={`font-mono text-[13px] ${className ?? ""}`}>
          {children}
        </code>
      );
    }
    return (
      <code className="rounded bg-[var(--cnrs-grey-light)] px-1 py-0.5 font-mono text-[12.5px] text-[var(--cnrs-blue)]">
        {children}
      </code>
    );
  },
  pre: ({ children }) => (
    <pre className="my-4 overflow-x-auto rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--bg-soft)] px-3 py-2.5 text-[13px] leading-relaxed">
      {children}
    </pre>
  ),
  table: ({ children }) => (
    <div className="my-3 overflow-x-auto rounded-[var(--radius-md)] border border-[var(--border)]">
      <table className="w-full text-sm">{children}</table>
    </div>
  ),
  thead: ({ children }) => (
    <thead className="border-b border-[var(--border)] bg-[var(--bg-soft)]">
      {children}
    </thead>
  ),
  th: ({ children }) => (
    <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="border-t border-[var(--border)] px-3 py-2 text-[14px] text-[var(--text-body)]">
      {children}
    </td>
  ),
  hr: () => <hr className="my-5 border-[var(--border)]" />,
};

// react-markdown's render is pure JS (no eval), so a SyntaxError thrown from
// inside it means a dependency chunk failed to parse — a Turbopack dev-mode
// hiccup (e.g. the page left open across a backend restart), not bad content.
// This boundary keeps that from crashing the whole chat view: it falls back to
// the raw answer text. A reload re-fetches the chunk and restores formatting.
class MarkdownBoundary extends Component<
  { fallback: ReactNode; children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

export function Markdown({
  children,
  className = "",
}: {
  children: string;
  className?: string;
}) {
  return (
    <div className={`max-w-none ${className}`}>
      <MarkdownBoundary
        fallback={
          <div className="whitespace-pre-wrap text-[15px] leading-relaxed text-[var(--text-body)]">
            {children}
          </div>
        }
      >
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeSanitize]}
          components={components}
        >
          {children}
        </ReactMarkdown>
      </MarkdownBoundary>
    </div>
  );
}
