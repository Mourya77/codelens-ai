import { CheckCircle2, Loader2, X, XCircle } from "lucide-react";

import type { IndexEvent } from "../types";

export interface IndexingState {
  repoUrl: string;
  progress: number;
  stage: string;
  message: string;
  totalChunks?: number;
  totalFiles?: number;
  done: boolean;
  error?: string;
  cached?: boolean;
}

// Maps raw SSE stages onto the four canonical pipeline stages from the
// spec: Fetching Tree -> AST Chunking -> Vector Embedding -> Ready.
const STAGE_STEPS = [
  { key: "fetch", label: "Fetching Tree" },
  { key: "chunk", label: "AST Chunking" },
  { key: "embed", label: "Vector Embedding" },
  { key: "ready", label: "Ready" },
] as const;

export function stageStepIndex(stage: string): number {
  switch (stage) {
    case "fetching_metadata":
    case "fetching_tree":
    case "fetching_files":
      return 0;
    case "chunking":
      return 1;
    case "embedding":
    case "storing":
      return 2;
    case "complete":
      return 3;
    default:
      return 0;
  }
}

interface IndexingBannerProps {
  state: IndexingState;
  onDismiss: () => void;
}

export default function IndexingBanner({ state, onDismiss }: IndexingBannerProps) {
  const activeStep = state.error
    ? stageStepIndex(state.stage)
    : state.done
      ? 3
      : stageStepIndex(state.stage);
  return (
    <div className="shrink-0 border-b border-ink-700 bg-ink-850 px-4 py-3">
      <div className="flex items-center gap-3">
        {state.error ? (
          <XCircle className="w-5 h-5 text-red-400 shrink-0" />
        ) : state.done ? (
          <CheckCircle2 className="w-5 h-5 text-lens shrink-0" />
        ) : (
          <Loader2 className="w-5 h-5 text-lens shrink-0 animate-spin" />
        )}
        <div className="min-w-0">
          <div className="text-sm text-slate-200 truncate">
            <span className="font-mono text-slate-400 mr-2">{state.repoUrl}</span>
            {state.error ? (
              <span className="text-red-300">{state.error}</span>
            ) : (
              state.message
            )}
          </div>
          <div className="flex items-center gap-2 mt-1.5">
            {STAGE_STEPS.map((step, i) => (
              <span key={step.key} className="flex items-center gap-2">
                <span
                  className={`text-[11px] font-medium px-2 py-0.5 rounded-full border ${
                    state.error && i === activeStep
                      ? "border-red-400/40 text-red-300 bg-red-400/10"
                      : i < activeStep || state.done
                        ? "border-lens/40 text-lens-soft bg-lens/10"
                        : i === activeStep
                          ? "border-lens/60 text-lens bg-lens/10"
                          : "border-ink-600 text-slate-500"
                  }`}
                >
                  {step.label}
                </span>
                {i < STAGE_STEPS.length - 1 && (
                  <span
                    className={`w-4 h-px ${i < activeStep || state.done ? "bg-lens/50" : "bg-ink-600"}`}
                  />
                )}
              </span>
            ))}
            {(state.totalChunks !== undefined || state.totalFiles !== undefined) && (
              <span className="text-[11px] font-mono text-slate-400 ml-2">
                {state.totalFiles !== undefined && `${state.totalFiles} files`}
                {state.totalFiles !== undefined && state.totalChunks !== undefined && " · "}
                {state.totalChunks !== undefined && `${state.totalChunks} chunks`}
                {state.cached && " · cached"}
              </span>
            )}
          </div>
        </div>
        <div className="flex-1" />
        <div className="font-mono text-sm text-slate-300 w-12 text-right">
          {Math.round(state.progress)}%
        </div>
        {(state.done || state.error) && (
          <button
            type="button"
            onClick={onDismiss}
            className="text-slate-500 hover:text-slate-300 p-1"
            aria-label="Dismiss"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>
      <div className="mt-2.5 h-1.5 rounded-full bg-ink-700 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            state.error ? "bg-red-400" : "bg-lens"
          }`}
          style={{ width: `${Math.min(100, Math.max(0, state.progress))}%` }}
        />
      </div>
    </div>
  );
}

export function eventToState(repoUrl: string, event: IndexEvent): IndexingState {
  return {
    repoUrl,
    progress: event.progress,
    stage: event.stage,
    message: event.message,
    totalChunks: typeof event.total_chunks === "number" ? event.total_chunks : undefined,
    totalFiles:
      typeof event.total_files === "number"
        ? event.total_files
        : typeof event.fetched_files === "number"
          ? event.fetched_files
          : undefined,
    done: event.stage === "complete",
    error: event.stage === "error" ? event.message : undefined,
    cached: event.cached === true,
  };
}
