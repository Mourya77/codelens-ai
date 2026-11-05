import { ExternalLink, Gauge, Hash, Layers, X } from "lucide-react";

import type { RepoSummary, SourceChunk } from "../types";
import CodeView from "./CodeView";

interface InspectorProps {
  chunk: SourceChunk | null;
  repo: RepoSummary | null;
  loading: boolean;
  onClose: () => void;
}

export function githubLineUrl(
  repo: RepoSummary | null,
  chunk: SourceChunk,
): string | null {
  if (!repo) return null;
  const ref = repo.commit_sha || repo.default_branch || "HEAD";
  return (
    `https://github.com/${repo.owner}/${repo.name}/blob/${ref}/` +
    `${chunk.file_path}#L${chunk.start_line}-L${chunk.end_line}`
  );
}

export default function Inspector({ chunk, repo, loading, onClose }: InspectorProps) {
  if (!chunk && !loading) return null;
  const ghUrl = chunk ? githubLineUrl(repo, chunk) : null;
  return (
    <aside className="drawer-enter w-full sm:w-[430px] shrink-0 border-l border-ink-700 bg-ink-900 flex flex-col min-h-0 max-h-full">
      <div className="flex items-center gap-2 px-4 py-3 border-b border-ink-700">
        <Layers className="w-4 h-4 text-lens" />
        <h2 className="text-sm font-semibold text-slate-100">
          Code Citation Inspector
        </h2>
        <div className="flex-1" />
        <button
          type="button"
          onClick={onClose}
          className="text-slate-500 hover:text-slate-300 p-1"
          aria-label="Close inspector"
        >
          <X className="w-4.5 h-4.5" size={18} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 min-h-0">
        {loading && <p className="text-sm text-slate-400">Loading chunk…</p>}
        {!loading && chunk && (
          <div className="space-y-3.5">
            <div>
              <div className="font-mono text-[13px] text-lens-soft break-all">
                {chunk.file_path}:{chunk.start_line}-{chunk.end_line}
              </div>
              <div className="text-xs text-slate-400 font-mono mt-1 break-all">
                {chunk.context_header}
              </div>
            </div>

            <div className="flex flex-wrap gap-1.5">
              {chunk.symbol_name && (
                <span className="inline-flex items-center gap-1 text-[11px] font-mono px-2 py-1 rounded-full bg-ink-700 text-slate-200">
                  <Hash className="w-3 h-3 text-lens" />
                  {chunk.symbol_name}
                  <span className="text-slate-500">({chunk.symbol_type})</span>
                </span>
              )}
              <span className="text-[11px] font-mono px-2 py-1 rounded-full bg-ink-700 text-slate-300">
                {chunk.language}
              </span>
              <span className="text-[11px] font-mono px-2 py-1 rounded-full bg-ink-700 text-slate-300">
                lines {chunk.start_line}–{chunk.end_line}
              </span>
              {typeof chunk.rrf_score === "number" && chunk.rrf_score > 0 && (
                <span className="inline-flex items-center gap-1 text-[11px] font-mono px-2 py-1 rounded-full bg-lens/10 border border-lens/30 text-lens-soft">
                  <Gauge className="w-3 h-3" />
                  RRF {chunk.rrf_score.toFixed(4)}
                  {typeof chunk.semantic_score === "number" &&
                    ` · cos ${chunk.semantic_score.toFixed(3)}`}
                  {typeof chunk.lexical_score === "number" &&
                    ` · lexical ${chunk.lexical_score.toFixed(2)}`}
                </span>
              )}
            </div>

            {chunk.imports && chunk.imports.length > 0 && (
              <div className="text-xs text-slate-400">
                <span className="text-slate-500 font-medium">Imports in file: </span>
                <span className="font-mono">{chunk.imports.join(", ")}</span>
              </div>
            )}

            <CodeView code={chunk.content} startLine={chunk.start_line} />

            {ghUrl && (
              <a
                href={ghUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1.5 text-sm text-sky-300 hover:text-sky-200 font-medium"
              >
                <ExternalLink className="w-4 h-4" />
                Open these exact lines on GitHub
              </a>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
