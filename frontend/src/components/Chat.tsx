import { useEffect, useRef, useState } from "react";
import {
  Bot,
  BrainCircuit,
  ChevronDown,
  ChevronRight,
  GitBranch,
  Lightbulb,
  Search,
  SendHorizonal,
  Sparkles,
  User,
} from "lucide-react";

import type { ChatTurn, CitationRef, SourceChunk, TraceStep } from "../types";
import Markdown from "./Markdown";

export const STARTER_QUESTIONS = [
  "Explain the high-level architecture and entry point",
  "How are errors and rate limits handled?",
  "Trace the request lifecycle",
];

function stageIcon(stage: string) {
  switch (stage) {
    case "analyze":
      return <Lightbulb className="w-3.5 h-3.5 text-amber-300" />;
    case "retrieve":
      return <Search className="w-3.5 h-3.5 text-sky-300" />;
    case "grade":
      return <BrainCircuit className="w-3.5 h-3.5 text-fuchsia-300" />;
    case "hop":
      return <GitBranch className="w-3.5 h-3.5 text-lens" />;
    case "answer":
      return <Sparkles className="w-3.5 h-3.5 text-lens" />;
    default:
      return <Bot className="w-3.5 h-3.5 text-slate-400" />;
  }
}

function TraceAccordion({ trace, hopsUsed }: { trace: TraceStep[]; hopsUsed?: number }) {
  const [open, setOpen] = useState(false);
  const hopCount = trace.filter((t) => t.stage === "hop").length;
  const graderWentSecondHop = hopCount > 0;
  return (
    <div className="mb-3 border border-ink-600 rounded-xl bg-ink-850 overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3.5 py-2.5 text-left hover:bg-ink-800/60 transition-colors"
      >
        {open ? (
          <ChevronDown className="w-4 h-4 text-slate-400" />
        ) : (
          <ChevronRight className="w-4 h-4 text-slate-400" />
        )}
        <BrainCircuit className="w-4 h-4 text-lens" />
        <span className="text-sm font-medium text-slate-200">
          Agent Reasoning Trace
        </span>
        <span
          className={`text-[11px] font-mono px-2 py-0.5 rounded-full border ${
            graderWentSecondHop
              ? "border-lens/40 text-lens bg-lens/10"
              : "border-ink-600 text-slate-400"
          }`}
        >
          {graderWentSecondHop
            ? `self-reflection triggered ${hopCount} second hop${hopCount > 1 ? "s" : ""}`
            : "grader accepted first retrieval"}
        </span>
        <span className="ml-auto text-[11px] font-mono text-slate-500">
          {trace.length} steps{(hopsUsed ?? 0) > 0 ? ` · ${hopsUsed} hop${(hopsUsed ?? 0) > 1 ? "s" : ""}` : ""}
        </span>
      </button>
      {open && (
        <ol className="px-3.5 py-3 space-y-2.5 border-t border-ink-700">
          {trace.map((step) => (
            <li key={step.step} className="flex gap-2.5 text-sm">
              <span className="mt-0.5 shrink-0 w-5 h-5 rounded-full bg-ink-700 flex items-center justify-center">
                {stageIcon(step.stage)}
              </span>
              <div className="min-w-0">
                <p className="text-slate-300 leading-snug">{step.description}</p>
                {typeof step.reasoning === "string" && step.reasoning && (
                  <p className="text-xs text-slate-500 mt-0.5 leading-snug">
                    {step.reasoning}
                  </p>
                )}
                {step.stage === "grade" && Array.isArray(step.missing) && step.missing.length > 0 && (
                  <p className="mt-1 flex flex-wrap gap-1">
                    {(step.missing as string[]).map((symbol) => (
                      <span
                        key={symbol}
                        className="font-mono text-[11px] px-1.5 py-px rounded bg-fuchsia-400/10 border border-fuchsia-400/30 text-fuchsia-200"
                      >
                        {symbol}
                      </span>
                    ))}
                  </p>
                )}
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function SourceCards({
  sources,
  onInspect,
}: {
  sources: SourceChunk[];
  onInspect: (chunk: SourceChunk) => void;
}) {
  if (sources.length === 0) return null;
  return (
    <div className="mt-3.5">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-1.5">
        Sources · hybrid RRF retrieval
      </div>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {sources.map((source) => (
          <button
            key={source.chunk_id}
            type="button"
            onClick={() => onInspect(source)}
            className="shrink-0 w-60 text-left bg-ink-850 hover:bg-ink-800 border border-ink-600 hover:border-lens/40 rounded-lg p-2.5 transition-colors"
          >
            <span className="block font-mono text-[11px] text-lens-soft truncate">
              {source.file_path}:{source.start_line}-{source.end_line}
            </span>
            <span className="block font-mono text-xs text-slate-200 truncate mt-1">
              {source.symbol_name ?? "(module header)"}
              <span className="text-slate-500"> · {source.symbol_type}</span>
            </span>
            {typeof source.rrf_score === "number" && (
              <span className="block text-[11px] font-mono text-slate-500 mt-1">
                RRF {source.rrf_score.toFixed(4)}
                {typeof source.semantic_score === "number" &&
                  ` · cos ${source.semantic_score.toFixed(2)}`}
                {typeof source.lexical_score === "number" &&
                  ` · bm25 ${source.lexical_score.toFixed(1)}`}
              </span>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}

interface ChatProps {
  turns: ChatTurn[];
  hasRepo: boolean;
  repoLabel: string;
  sending: boolean;
  onSend: (question: string) => void;
  onCitation: (ref: CitationRef, turn: ChatTurn) => void;
  onInspectSource: (chunk: SourceChunk) => void;
}

export default function Chat({
  turns,
  hasRepo,
  repoLabel,
  sending,
  onSend,
  onCitation,
  onInspectSource,
}: ChatProps) {
  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns]);

  const submit = (value: string) => {
    const trimmed = value.trim();
    if (!trimmed || sending || !hasRepo) return;
    onSend(trimmed);
    setDraft("");
  };

  return (
    <section className="flex-1 flex flex-col min-w-0 min-h-0 bg-ink-950">
      <div ref={scrollRef} className="flex-1 overflow-y-auto min-h-0">
        <div className="max-w-3xl mx-auto px-5 py-6 space-y-7">
          {turns.length === 0 && (
            <div className="pt-10 text-center">
              <Bot className="w-10 h-10 text-lens mx-auto" />
              <h2 className="text-xl font-semibold text-slate-100 mt-4">
                Ask the codebase anything
              </h2>
              <p className="text-sm text-slate-400 mt-2 max-w-md mx-auto leading-relaxed">
                {hasRepo ? (
                  <>
                    <span className="font-mono text-lens-soft">{repoLabel}</span>{" "}
                    is indexed. The agent plans a search, grades its own
                    retrieved context, and follows missing symbols on its own —
                    every claim cites exact lines.
                  </>
                ) : (
                  "Index a repository first — then the agent can plan, retrieve, self-grade and answer with line-level citations."
                )}
              </p>
              {hasRepo && (
                <div className="flex flex-wrap justify-center gap-2 mt-6">
                  {STARTER_QUESTIONS.map((question) => (
                    <button
                      key={question}
                      type="button"
                      onClick={() => submit(question)}
                      className="text-sm bg-ink-850 hover:bg-ink-800 border border-ink-600 hover:border-lens/40 text-slate-200 rounded-full px-3.5 py-2 transition-colors"
                    >
                      {question}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {turns.map((turn) =>
            turn.role === "user" ? (
              <div key={turn.id} className="flex gap-3">
                <span className="w-7 h-7 rounded-full bg-ink-700 flex items-center justify-center shrink-0">
                  <User className="w-4 h-4 text-slate-300" />
                </span>
                <div className="pt-1 text-[15px] text-slate-100 whitespace-pre-wrap">
                  {turn.content}
                </div>
              </div>
            ) : (
              <div key={turn.id} className="flex gap-3">
                <span className="w-7 h-7 rounded-full bg-lens/15 border border-lens/30 flex items-center justify-center shrink-0">
                  <Bot className="w-4 h-4 text-lens" />
                </span>
                <div className="flex-1 min-w-0 pt-0.5">
                  {turn.pending ? (
                    <div className="flex items-center gap-2 text-sm text-slate-400 py-2">
                      <span className="flex gap-1">
                        {[0, 1, 2].map((i) => (
                          <span
                            key={i}
                            className="w-1.5 h-1.5 rounded-full bg-lens agent-working-dot"
                            style={{ animationDelay: `${i * 0.18}s` }}
                          />
                        ))}
                      </span>
                      Agent is planning, retrieving and grading its context…
                    </div>
                  ) : turn.error ? (
                    <div className="text-sm text-red-300 bg-red-400/10 border border-red-400/30 rounded-lg px-3.5 py-2.5">
                      {turn.error}
                    </div>
                  ) : (
                    <>
                      {turn.trace && turn.trace.length > 0 && (
                        <TraceAccordion trace={turn.trace} hopsUsed={turn.hopsUsed} />
                      )}
                      <Markdown
                        text={turn.content}
                        onCitation={(ref) => onCitation(ref, turn)}
                      />
                      {turn.sources && (
                        <SourceCards sources={turn.sources} onInspect={onInspectSource} />
                      )}
                    </>
                  )}
                </div>
              </div>
            ),
          )}
        </div>
      </div>

      <div className="shrink-0 border-t border-ink-700 bg-ink-900 px-4 py-3.5">
        <form
          className="max-w-3xl mx-auto flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            submit(draft);
          }}
        >
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            disabled={!hasRepo || sending}
            placeholder={
              hasRepo
                ? "Ask about a function, a file, a flow… (try naming an exact symbol)"
                : "Index a repository to start chatting"
            }
            className="flex-1 bg-ink-850 border border-ink-600 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-lens/60 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={!hasRepo || sending || !draft.trim()}
            className="inline-flex items-center gap-1.5 bg-lens text-ink-950 font-semibold text-sm rounded-xl px-4 py-3 hover:bg-lens-soft disabled:opacity-40 transition-colors"
          >
            <SendHorizonal className="w-4 h-4" />
            Ask
          </button>
        </form>
        <p className="max-w-3xl mx-auto text-[11px] text-slate-600 mt-2 font-mono">
          agentic rag · hybrid semantic + BM25 retrieval · RRF fusion ·
          self-reflection grader · max 2 hops
        </p>
      </div>
    </section>
  );
}
