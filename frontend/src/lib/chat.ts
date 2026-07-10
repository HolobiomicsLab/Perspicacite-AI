// Streaming chat client for POST /api/chat (stream=true).
//
// The Perspicacité backend emits an SSE-style line stream. Most frames
// look like:
//   data: {"token":"..."}
//   data: {"papers_found": 3, "sources":[...]}
//   data: {"kind":"provider_progress","phase":"start","providers":["semantic_scholar",...]}
//   data: {"kind":"batch_progress","stage":"abstract_analysis","current":4,"total":10}
//   data: {"kind":"query_rephrased","original":"...","rephrased":"..."}
//   data: {"message":"Generating answer…"}
//   data: {"answer":"...","conversation_id":"..."}
//
// We parse defensively — anything with a `data:` line that's not JSON
// gets treated as a raw text token.

import type { RAGMode } from "./modes";
import type { DatabaseId } from "./databases";

export type ChatSource = {
  paper_id?: string;
  title?: string;
  authors?: string[];
  year?: number;
  doi?: string;
  url?: string;
  journal?: string;
  abstract?: string;
  relevance_score?: number;
  citation_count?: number;
  provider?: string;
  providers?: string[];
  discovery_sources?: string[];
  enrichment_sources?: string[];
  oa_url?: string | null;
  pdf_url?: string | null;
};

export type ThinkingStepKind =
  | "status"
  | "query_rephrased"
  | "provider_progress"
  | "batch_progress";

export type ThinkingStep = {
  id: string;
  kind: ThinkingStepKind;
  label: string;
  // Optional structured payload; renderer decides what to show.
  detail?: {
    original?: string;
    rephrased?: string;
    providers?: string[];
    byProvider?: Record<string, number>;
    phase?: "start" | "done";
    stage?: string;
    current?: number;
    total?: number;
  };
  ts: number;
};

export type ChatStreamEvent =
  | { kind: "token"; text: string }
  | { kind: "meta"; papers_found?: number; sources?: ChatSource[] }
  | { kind: "thinking"; step: ThinkingStep }
  | { kind: "done"; conversation_id?: string; answer?: string }
  | { kind: "error"; message: string }
  | { kind: "metadata"; iteration_count: number; completion_reason: string; diagnostic: Record<string, unknown> | null };

// Empty string → same-origin requests, proxied by Next.js rewrites in
// next.config.ts. Override via NEXT_PUBLIC_PERSPICACITE_URL for direct
// cross-origin calls (requires backend CORS).
const BACKEND = process.env.NEXT_PUBLIC_PERSPICACITE_URL ?? "";

let _nidCounter = 0;
function nid(): string {
  return `step-${Date.now().toString(36)}-${(_nidCounter++).toString(36)}`;
}

export async function* streamChat(opts: {
  query: string;
  mode: RAGMode;
  kbName?: string;
  conversationId?: string;
  maxPapers?: number;
  databases?: DatabaseId[];
  bm25Weight?: number;
  vectorWeight?: number;
  signal?: AbortSignal;
}): AsyncGenerator<ChatStreamEvent> {
  // Drop kb_name when empty so the backend default kicks in.
  const body: Record<string, unknown> = {
    query: opts.query,
    mode: opts.mode,
    stream: true,
    max_papers: opts.maxPapers ?? 10,
    conversation_id: opts.conversationId,
    databases: opts.databases,
  };
  if (opts.bm25Weight !== undefined) body.bm25_weight = opts.bm25Weight;
  if (opts.vectorWeight !== undefined) body.vector_weight = opts.vectorWeight;
  if (opts.kbName) body.kb_name = opts.kbName;

  const res = await fetch(`${BACKEND}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal: opts.signal,
  });

  if (!res.ok || !res.body) {
    yield { kind: "error", message: `HTTP ${res.status}` };
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by blank lines.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      for (const ev of parseFrame(frame)) yield ev;
    }
  }

  if (buffer.trim()) {
    for (const ev of parseFrame(buffer)) yield ev;
  }
}

function parseFrame(frame: string): ChatStreamEvent[] {
  const dataLine = frame
    .split("\n")
    .find((l) => l.trim().toLowerCase().startsWith("data:"));
  if (!dataLine) return [];
  const json = dataLine.replace(/^data:\s*/i, "").trim();
  if (!json || json === "[DONE]") return [{ kind: "done" }];

  let obj: Record<string, unknown>;
  try {
    obj = JSON.parse(json);
  } catch {
    return [{ kind: "token", text: json }];
  }

  // Token deltas — the backend uses `{"type":"token","delta":"..."}`
  // (or `delta_b64` for Unicode-safe base64). Older / future variants
  // may put the token directly under `token`, `chunk`, or `delta`.
  if (typeof obj.token === "string") return [{ kind: "token", text: obj.token }];
  if (typeof obj.chunk === "string") return [{ kind: "token", text: obj.chunk }];
  if (obj.type === "token") {
    let text = "";
    if (typeof obj.delta === "string") text = obj.delta;
    else if (typeof obj.delta_b64 === "string") text = decodeB64Utf8(obj.delta_b64);
    if (text) return [{ kind: "token", text }];
  }
  if (typeof obj.delta === "string") return [{ kind: "token", text: obj.delta }];

  // Final-answer payload — some modes (basic, contradiction, deep_research)
  // ship the full text in one `type:"answer"` frame instead of streaming
  // token by token. Mark this as `done` with `answer` so ChatPanel will
  // adopt it as the message body only when no streamed text is present.
  if (obj.type === "answer") {
    const text =
      typeof obj.content === "string"
        ? obj.content
        : typeof obj.content_b64 === "string"
          ? decodeB64Utf8(obj.content_b64)
          : "";
    if (text) {
      return [
        {
          kind: "done",
          answer: text,
          conversation_id:
            typeof obj.conversation_id === "string" ? obj.conversation_id : undefined,
        },
      ];
    }
  }

  // papers_found events ship a richer list of papers; treat as meta.
  if (obj.type === "papers_found" && Array.isArray(obj.papers)) {
    return [
      {
        kind: "meta",
        papers_found: obj.papers.length,
        sources: obj.papers as ChatSource[],
      },
    ];
  }

  if (obj.type === "metadata") {
    return [
      {
        kind: "metadata" as const,
        iteration_count: typeof obj.iteration_count === "number" ? obj.iteration_count : 1,
        completion_reason: typeof obj.completion_reason === "string" ? obj.completion_reason : "complete",
        diagnostic:
          obj.diagnostic && typeof obj.diagnostic === "object" && !Array.isArray(obj.diagnostic)
            ? (obj.diagnostic as Record<string, unknown>)
            : null,
      },
    ];
  }

  // Source events appear during streaming for some modes — collect.
  if (obj.type === "source" && obj.source) {
    return [
      {
        kind: "meta",
        sources: [obj.source as ChatSource],
      },
    ];
  }

  // Structured thinking-step events.
  if (obj.kind === "query_rephrased") {
    // Backend field is `rewritten`; older variants use `rephrased` — accept both.
    const rephrased =
      typeof obj.rewritten === "string"
        ? (obj.rewritten as string)
        : typeof obj.rephrased === "string"
          ? (obj.rephrased as string)
          : undefined;
    return [
      {
        kind: "thinking",
        step: {
          id: nid(),
          kind: "query_rephrased",
          label: "Query rephrased",
          detail: {
            original: typeof obj.original === "string" ? obj.original : undefined,
            rephrased,
          },
          ts: Date.now(),
        },
      },
    ];
  }

  // Agentic-mode specific events. We surface intent_result and
  // phase_transition as status steps so the reasoning trail stays
  // intelligible even when the backend emits rich structured frames.
  if (obj.type === "intent_result" && typeof obj.intent === "string") {
    const reasoning =
      typeof obj.reasoning === "string" ? ` — ${obj.reasoning}` : "";
    return [
      {
        kind: "thinking",
        step: {
          id: nid(),
          kind: "status",
          label: `Intent: ${obj.intent}${reasoning}`,
          ts: Date.now(),
        },
      },
    ];
  }
  if (obj.type === "phase_transition" && typeof obj.phase === "string") {
    return [
      {
        kind: "thinking",
        step: {
          id: nid(),
          kind: "status",
          label: `Phase → ${obj.phase}`,
          ts: Date.now(),
        },
      },
    ];
  }
  if (obj.type === "thinking" && typeof obj.message === "string") {
    return [
      {
        kind: "thinking",
        step: {
          id: nid(),
          kind: "status",
          label: obj.message,
          ts: Date.now(),
        },
      },
    ];
  }
  // Plain status frames carry the live phase text (e.g. "Basic RAG:
  // Retrieving documents…", "Basic RAG: Generating response…"). Surface
  // them as steps so the agentic trail advances through its phases in real
  // time as the answer streams, instead of only snapping to done at the end.
  if (obj.type === "status" && typeof obj.message === "string") {
    return [
      {
        kind: "thinking",
        step: {
          id: nid(),
          kind: "status",
          label: obj.message,
          ts: Date.now(),
        },
      },
    ];
  }
  if (obj.kind === "provider_progress") {
    const phase = obj.phase === "done" ? "done" : "start";
    const providers = Array.isArray(obj.providers)
      ? (obj.providers as string[])
      : undefined;
    const byProvider =
      obj.by_provider && typeof obj.by_provider === "object"
        ? (obj.by_provider as Record<string, number>)
        : undefined;
    return [
      {
        kind: "thinking",
        step: {
          id: nid(),
          kind: "provider_progress",
          label:
            phase === "start"
              ? "Querying databases"
              : "Database results",
          detail: { phase, providers, byProvider },
          ts: Date.now(),
        },
      },
    ];
  }
  if (obj.kind === "batch_progress") {
    const stage = typeof obj.stage === "string" ? obj.stage : "batch";
    const current = typeof obj.current === "number" ? obj.current : 0;
    const total = typeof obj.total === "number" ? obj.total : 0;
    const stageLabel: Record<string, string> = {
      abstract_analysis: "Abstract analysis",
      theme_assignment: "Theme assignment",
      pdf_download: "Downloading PDFs",
      chunking: "Chunking",
      embedding: "Embedding",
    };
    return [
      {
        kind: "thinking",
        step: {
          id: nid(),
          kind: "batch_progress",
          label: stageLabel[stage] ?? stage.replace(/_/g, " "),
          detail: { stage, current, total },
          ts: Date.now(),
        },
      },
    ];
  }
  if (typeof obj.message === "string") {
    return [
      {
        kind: "thinking",
        step: {
          id: nid(),
          kind: "status",
          label: obj.message,
          ts: Date.now(),
        },
      },
    ];
  }

  // Final answer w/ sources.
  if (typeof obj.answer === "string" && obj.sources)
    return [
      {
        kind: "done",
        answer: obj.answer,
        conversation_id:
          typeof obj.conversation_id === "string" ? obj.conversation_id : undefined,
      },
    ];

  // Meta (sources / papers_found).
  if (obj.sources || typeof obj.papers_found === "number") {
    return [
      {
        kind: "meta",
        papers_found: typeof obj.papers_found === "number" ? obj.papers_found : undefined,
        sources: Array.isArray(obj.sources) ? (obj.sources as ChatSource[]) : undefined,
      },
    ];
  }

  if (obj.event === "done" || obj.done === true)
    return [
      {
        kind: "done",
        conversation_id:
          typeof obj.conversation_id === "string" ? obj.conversation_id : undefined,
      },
    ];

  return [];
}

export async function cancelChat(conversationId: string): Promise<void> {
  try {
    await fetch(`${BACKEND}/api/chat/cancel`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation_id: conversationId }),
    });
  } catch {
    // Best-effort.
  }
}

// Unicode-safe base64 → string. Used for `delta_b64` / `content_b64`
// frames the backend uses for tokens that contain multi-byte characters.
function decodeB64Utf8(b64: string): string {
  try {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new TextDecoder("utf-8").decode(bytes);
  } catch {
    return "";
  }
}

// ─── Token-estimation helpers ────────────────────────────────────────────
// Rough 4-chars-per-token approximation matches the heuristic in the
// original GUI's chat.js so the displayed counts stay comparable.

export function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.max(1, Math.round(text.length / 4));
}
