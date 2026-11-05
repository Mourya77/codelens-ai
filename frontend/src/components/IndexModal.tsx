import { useState } from "react";
import { FolderGit2, Link2, Rocket, X, Zap } from "lucide-react";

export const SAMPLE_REPOS = [
  {
    label: "tiangolo/fastapi",
    url: "https://github.com/tiangolo/fastapi",
    note: "The famous Python API framework — great architecture questions",
    icon: "rocket" as const,
  },
  {
    label: "octocat/Hello-World",
    url: "https://github.com/octocat/Hello-World",
    note: "Tiny demo repo — indexes in seconds, instant gratification",
    icon: "zap" as const,
  },
];

interface IndexModalProps {
  open: boolean;
  indexing: boolean;
  onClose: () => void;
  onSubmit: (repoUrl: string) => void;
}

export default function IndexModal({
  open,
  indexing,
  onClose,
  onSubmit,
}: IndexModalProps) {
  const [url, setUrl] = useState("");
  if (!open) return null;

  const submit = (value: string) => {
    const trimmed = value.trim();
    if (trimmed) onSubmit(trimmed);
  };

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-[2px] flex items-start justify-center pt-[12vh] px-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-xl bg-ink-850 border border-ink-600 rounded-2xl shadow-panel p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
              <FolderGit2 className="w-5 h-5 text-lens" />
              Index a GitHub repository
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              CodeLens fetches the repo in memory (no clone), chunks it with an
              AST-aware parser, and embeds every chunk for hybrid search.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-slate-500 hover:text-slate-300 p-1"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <form
          className="mt-5"
          onSubmit={(e) => {
            e.preventDefault();
            submit(url);
          }}
        >
          <label className="text-xs font-medium uppercase tracking-wide text-slate-500">
            GitHub URL
          </label>
          <div className="mt-1.5 flex gap-2">
            <div className="relative flex-1">
              <Link2 className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                autoFocus
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://github.com/owner/repo"
                className="w-full bg-ink-900 border border-ink-600 rounded-lg pl-9 pr-3 py-2.5 text-sm font-mono text-slate-100 placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-lens/60"
              />
            </div>
            <button
              type="submit"
              disabled={indexing || !url.trim()}
              className="bg-lens text-ink-950 font-semibold text-sm rounded-lg px-4 py-2.5 hover:bg-lens-soft disabled:opacity-40 transition-colors"
            >
              {indexing ? "Indexing…" : "Index"}
            </button>
          </div>
        </form>

        <div className="mt-6">
          <div className="text-xs font-medium uppercase tracking-wide text-slate-500 mb-2">
            Try a sample repo
          </div>
          <div className="grid sm:grid-cols-2 gap-2.5">
            {SAMPLE_REPOS.map((sample) => (
              <button
                key={sample.url}
                type="button"
                disabled={indexing}
                onClick={() => submit(sample.url)}
                className="text-left bg-ink-800 hover:bg-ink-750 border border-ink-600 hover:border-lens/40 rounded-xl p-3.5 transition-colors disabled:opacity-40"
              >
                <span className="flex items-center gap-2 font-mono text-sm text-slate-100">
                  {sample.icon === "rocket" ? (
                    <Rocket className="w-4 h-4 text-lens" />
                  ) : (
                    <Zap className="w-4 h-4 text-lens" />
                  )}
                  {sample.label}
                </span>
                <span className="block text-xs text-slate-400 mt-1.5 leading-relaxed">
                  {sample.note}
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
