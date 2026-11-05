// API layer — every call is a relative /api path so the SPA and the
// FastAPI backend work from the same origin in production (Vite
// proxies /api to localhost:8000 in dev).

import type {
  IndexEvent,
  QueryResponse,
  RepoSummary,
  RepoTree,
  SourceChunk,
} from "./types";

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, init);
  if (!resp.ok) {
    throw new Error(`${resp.status} ${resp.statusText} — ${url}`);
  }
  return (await resp.json()) as T;
}

export function fetchRepos(): Promise<RepoSummary[]> {
  return jsonFetch<RepoSummary[]>("/api/repos");
}

export function fetchRepoTree(repoId: string): Promise<RepoTree> {
  return jsonFetch<RepoTree>(`/api/repos/${repoId}/tree`);
}

export async function fetchFileChunks(
  repoId: string,
  filePath: string,
): Promise<SourceChunk[]> {
  const data = await jsonFetch<{ chunks: SourceChunk[] }>(
    `/api/repos/${repoId}/chunks?file_path=${encodeURIComponent(filePath)}`,
  );
  return data.chunks;
}

export function queryRepo(
  repoId: string,
  question: string,
  chatHistory: { role: string; content: string }[],
): Promise<QueryResponse> {
  return jsonFetch<QueryResponse>(`/api/repos/${repoId}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, chat_history: chatHistory }),
  });
}

/**
 * Consume the indexing SSE stream. fetch() is used instead of
 * EventSource because the endpoint is a POST. Resolves with the final
 * event (stage "complete" or "error").
 */
export async function indexRepoStream(
  repoUrl: string,
  onEvent: (event: IndexEvent) => void,
  signal?: AbortSignal,
): Promise<IndexEvent> {
  const resp = await fetch("/api/repos/index", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo_url: repoUrl }),
    signal,
  });
  if (!resp.ok || !resp.body) {
    throw new Error(`Indexing request failed: ${resp.status}`);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalEvent: IndexEvent = {
    stage: "error",
    progress: 0,
    message: "Stream ended without a completion event.",
  };

  const handleBlock = (block: string) => {
    for (const line of block.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      try {
        const event = JSON.parse(trimmed.slice(5).trim()) as IndexEvent;
        finalEvent = event;
        onEvent(event);
      } catch {
        // Ignore malformed/partial events; the next block may complete.
      }
    }
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";
    for (const block of blocks) handleBlock(block);
  }
  if (buffer.trim()) handleBlock(buffer);
  return finalEvent;
}
