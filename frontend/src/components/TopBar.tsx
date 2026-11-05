import { ChevronDown, Database, Plus, ScanSearch } from "lucide-react";

import type { RepoSummary } from "../types";

interface TopBarProps {
  repos: RepoSummary[];
  activeRepo: RepoSummary | null;
  onSelectRepo: (repoId: string) => void;
  onIndexNew: () => void;
}

export default function TopBar({
  repos,
  activeRepo,
  onSelectRepo,
  onIndexNew,
}: TopBarProps) {
  return (
    <header className="h-14 shrink-0 flex items-center gap-4 px-4 border-b border-ink-700 bg-ink-900">
      <div className="flex items-center gap-2.5 select-none">
        <span className="w-8 h-8 rounded-lg bg-lens/15 border border-lens/30 flex items-center justify-center">
          <ScanSearch className="w-4.5 h-4.5 text-lens" size={18} />
        </span>
        <div className="leading-tight">
          <div className="font-semibold text-slate-100 tracking-tight">
            CodeLens <span className="text-lens font-mono">//</span> Agentic RAG
          </div>
          <div className="text-[11px] text-slate-500 font-mono">
            github repository &amp; code documentation analyzer
          </div>
        </div>
      </div>

      <div className="h-6 w-px bg-ink-600 mx-1 hidden sm:block" />

      <div className="relative flex items-center min-w-0">
        <Database className="w-4 h-4 text-slate-500 absolute left-2.5 pointer-events-none" />
        <select
          value={activeRepo?.id ?? ""}
          onChange={(e) => onSelectRepo(e.target.value)}
          disabled={repos.length === 0}
          className="appearance-none bg-ink-800 border border-ink-600 hover:border-ink-600/80 rounded-lg pl-9 pr-8 py-2 text-sm text-slate-200 max-w-[340px] truncate disabled:opacity-50 focus:outline-none focus:ring-1 focus:ring-lens/60 cursor-pointer"
          aria-label="Active repository"
        >
          {repos.length === 0 && <option value="">No repositories indexed</option>}
          {repos.map((repo) => (
            <option key={repo.id} value={repo.id}>
              {repo.owner}/{repo.name} · {repo.total_chunks} chunks
            </option>
          ))}
        </select>
        <ChevronDown className="w-4 h-4 text-slate-500 absolute right-2.5 pointer-events-none" />
      </div>

      {activeRepo && (
        <span
          className={`hidden md:inline-flex items-center gap-1.5 text-[11px] font-mono px-2 py-1 rounded-full border ${
            activeRepo.status === "ready"
              ? "text-lens-soft border-lens/30 bg-lens/10"
              : "text-amber-300 border-amber-400/30 bg-amber-400/10"
          }`}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              activeRepo.status === "ready" ? "bg-lens" : "bg-amber-300"
            }`}
          />
          {activeRepo.status} · {(activeRepo.commit_sha ?? "").slice(0, 7)}
        </span>
      )}

      <div className="flex-1" />

      <button
        type="button"
        onClick={onIndexNew}
        className="inline-flex items-center gap-1.5 bg-lens text-ink-950 font-semibold text-sm rounded-lg px-3.5 py-2 hover:bg-lens-soft transition-colors"
      >
        <Plus className="w-4 h-4" />
        Index New Repo
      </button>
    </header>
  );
}
