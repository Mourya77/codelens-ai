import { useCallback, useEffect, useRef, useState } from "react";

import {
  fetchFileChunks,
  fetchRepos,
  fetchRepoTree,
  indexRepoStream,
  queryRepo,
} from "./api";
import Chat from "./components/Chat";
import IndexModal from "./components/IndexModal";
import IndexingBanner, {
  eventToState,
  type IndexingState,
} from "./components/IndexingBanner";
import Inspector from "./components/Inspector";
import Sidebar from "./components/Sidebar";
import TopBar from "./components/TopBar";
import type {
  ChatTurn,
  CitationRef,
  FileEntry,
  IndexEvent,
  RepoSummary,
  SourceChunk,
} from "./types";

let turnCounter = 0;
const nextTurnId = () => `turn-${++turnCounter}`;

export default function App() {
  const [repos, setRepos] = useState<RepoSummary[]>([]);
  const [activeRepoId, setActiveRepoId] = useState<string | null>(null);
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [treeTotals, setTreeTotals] = useState({ files: 0, chunks: 0 });

  const [indexModalOpen, setIndexModalOpen] = useState(false);
  const [indexing, setIndexing] = useState<IndexingState | null>(null);
  const indexingRef = useRef(false);

  const [selectedFile, setSelectedFile] = useState<FileEntry | null>(null);
  const [fileChunks, setFileChunks] = useState<SourceChunk[] | null>(null);
  const [fileChunksLoading, setFileChunksLoading] = useState(false);

  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [sending, setSending] = useState(false);

  const [inspectorChunk, setInspectorChunk] = useState<SourceChunk | null>(null);
  const [inspectorLoading, setInspectorLoading] = useState(false);

  const activeRepo = repos.find((r) => r.id === activeRepoId) ?? null;

  const loadRepos = useCallback(async (selectId?: string) => {
    const list = await fetchRepos();
    setRepos(list);
    if (selectId) {
      setActiveRepoId(selectId);
    } else if (list.length > 0) {
      setActiveRepoId((current) =>
        current && list.some((r) => r.id === current) ? current : list[0].id,
      );
    }
    return list;
  }, []);

  useEffect(() => {
    loadRepos().catch(() => setRepos([]));
  }, [loadRepos]);

  // Load the tree whenever the active repo changes; reset per-repo UI.
  useEffect(() => {
    setSelectedFile(null);
    setFileChunks(null);
    setTurns([]);
    setInspectorChunk(null);
    if (!activeRepoId) {
      setFiles([]);
      setTreeTotals({ files: 0, chunks: 0 });
      return;
    }
    let cancelled = false;
    fetchRepoTree(activeRepoId)
      .then((tree) => {
        if (cancelled) return;
        setFiles(tree.files);
        setTreeTotals({ files: tree.total_files, chunks: tree.total_chunks });
      })
      .catch(() => {
        if (!cancelled) {
          setFiles([]);
          setTreeTotals({ files: 0, chunks: 0 });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [activeRepoId]);

  // --- indexing (SSE) -------------------------------------------------------

  const handleIndexSubmit = useCallback(
    async (repoUrl: string) => {
      setIndexModalOpen(false);
      if (indexingRef.current) return;
      indexingRef.current = true;
      setIndexing({
        repoUrl,
        progress: 0,
        stage: "starting",
        message: "Contacting GitHub…",
        done: false,
      });
      try {
        const finalEvent = await indexRepoStream(repoUrl, (event: IndexEvent) => {
          setIndexing((prev) => {
            const next = eventToState(repoUrl, event);
            // Keep the best-known counters — later events don't all
            // repeat them.
            return {
              ...next,
              totalChunks: next.totalChunks ?? prev?.totalChunks,
              totalFiles: next.totalFiles ?? prev?.totalFiles,
            };
          });
        });
        if (finalEvent.stage === "complete" && finalEvent.repo_id) {
          await loadRepos(finalEvent.repo_id);
        } else if (finalEvent.stage === "complete") {
          await loadRepos();
        }
      } catch (error) {
        setIndexing({
          repoUrl,
          progress: 0,
          stage: "error",
          message: "",
          done: false,
          error:
            error instanceof Error
              ? error.message
              : "Indexing failed unexpectedly.",
        });
      } finally {
        indexingRef.current = false;
      }
    },
    [loadRepos],
  );

  // --- sidebar ---------------------------------------------------------------

  const handleSelectFile = useCallback(
    async (file: FileEntry) => {
      setSelectedFile(file);
      if (!activeRepoId) return;
      setFileChunksLoading(true);
      try {
        setFileChunks(await fetchFileChunks(activeRepoId, file.file_path));
      } catch {
        setFileChunks([]);
      } finally {
        setFileChunksLoading(false);
      }
    },
    [activeRepoId],
  );

  // --- inspector ---------------------------------------------------------------

  const resolveCitation = useCallback(
    async (ref: CitationRef, turn?: ChatTurn): Promise<SourceChunk | null> => {
      const fromTurn = turn?.sources?.find(
        (s) =>
          s.file_path === ref.filePath &&
          s.start_line === ref.startLine &&
          s.end_line === ref.endLine,
      );
      if (fromTurn) return fromTurn;
      if (fileChunks) {
        const local = fileChunks.find(
          (s) =>
            s.file_path === ref.filePath &&
            s.start_line === ref.startLine &&
            s.end_line === ref.endLine,
        );
        if (local) return local;
      }
      if (!activeRepoId) return null;
      const chunks = await fetchFileChunks(activeRepoId, ref.filePath);
      return (
        chunks.find(
          (s) => s.start_line === ref.startLine && s.end_line === ref.endLine,
        ) ??
        chunks.find(
          (s) => ref.startLine >= s.start_line && ref.endLine <= s.end_line,
        ) ??
        null
      );
    },
    [activeRepoId, fileChunks],
  );

  const handleCitation = useCallback(
    (ref: CitationRef, turn: ChatTurn) => {
      setInspectorLoading(true);
      resolveCitation(ref, turn)
        .then((chunk) => {
          if (chunk) {
            setInspectorChunk(chunk);
          } else {
            setInspectorChunk({
              chunk_id: `unresolved-${ref.filePath}-${ref.startLine}`,
              file_path: ref.filePath,
              language: "text",
              symbol_name: null,
              symbol_type: "unknown",
              start_line: ref.startLine,
              end_line: ref.endLine,
              content:
                "// This citation's chunk was not among the retrieved " +
                "sources for this answer.",
              context_header: `File: ${ref.filePath}`,
            });
          }
        })
        .catch(() => setInspectorChunk(null))
        .finally(() => setInspectorLoading(false));
    },
    [resolveCitation],
  );

  // --- chat -----------------------------------------------------------------

  const handleSend = useCallback(
    async (question: string) => {
      if (!activeRepoId || sending) return;
      const history = turns.map((t) => ({ role: t.role, content: t.content }));
      const userTurn: ChatTurn = {
        id: nextTurnId(),
        role: "user",
        content: question,
      };
      const pendingId = nextTurnId();
      setTurns((prev) => [
        ...prev,
        userTurn,
        { id: pendingId, role: "assistant", content: "", pending: true },
      ]);
      setSending(true);
      try {
        const result = await queryRepo(activeRepoId, question, history);
        setTurns((prev) =>
          prev.map((t) =>
            t.id === pendingId
              ? {
                  id: pendingId,
                  role: "assistant",
                  content: result.answer,
                  trace: result.agent_trace,
                  sources: result.sources,
                  hopsUsed: result.hops_used,
                }
              : t,
          ),
        );
      } catch (error) {
        setTurns((prev) =>
          prev.map((t) =>
            t.id === pendingId
              ? {
                  id: pendingId,
                  role: "assistant",
                  content: "",
                  error:
                    error instanceof Error
                      ? `The agent couldn't answer: ${error.message}`
                      : "The agent couldn't answer that question.",
                }
              : t,
          ),
        );
      } finally {
        setSending(false);
      }
    },
    [activeRepoId, sending, turns],
  );

  return (
    <div className="h-full flex flex-col bg-ink-950">
      <TopBar
        repos={repos}
        activeRepo={activeRepo}
        onSelectRepo={setActiveRepoId}
        onIndexNew={() => setIndexModalOpen(true)}
      />
      {indexing && (
        <IndexingBanner state={indexing} onDismiss={() => setIndexing(null)} />
      )}
      <div className="flex-1 flex min-h-0">
        <Sidebar
          files={files}
          totalFiles={treeTotals.files}
          totalChunks={treeTotals.chunks}
          selectedFile={selectedFile}
          fileChunks={fileChunks}
          fileChunksLoading={fileChunksLoading}
          onSelectFile={handleSelectFile}
          onInspectChunk={setInspectorChunk}
        />
        <Chat
          turns={turns}
          hasRepo={!!activeRepo}
          repoLabel={activeRepo ? `${activeRepo.owner}/${activeRepo.name}` : ""}
          sending={sending}
          onSend={handleSend}
          onCitation={handleCitation}
          onInspectSource={setInspectorChunk}
        />
        <Inspector
          chunk={inspectorChunk}
          repo={activeRepo}
          loading={inspectorLoading}
          onClose={() => setInspectorChunk(null)}
        />
      </div>
      <IndexModal
        open={indexModalOpen}
        indexing={!!indexing && !indexing.done && !indexing.error}
        onClose={() => setIndexModalOpen(false)}
        onSubmit={handleIndexSubmit}
      />
    </div>
  );
}
