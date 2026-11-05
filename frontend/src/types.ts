// Types mirroring the CodeLens AI backend API (Phases 1–3).

export interface RepoSummary {
  id: string;
  repo_url: string;
  owner: string;
  name: string;
  default_branch: string;
  commit_sha: string | null;
  status: "pending" | "indexing" | "ready" | "failed";
  total_files: number;
  total_chunks: number;
  created_at: string | null;
}

export interface FileEntry {
  file_path: string;
  language: string;
  chunks: number;
}

export interface RepoTree {
  repo_id: string;
  owner: string;
  name: string;
  commit_sha: string | null;
  total_files: number;
  total_chunks: number;
  files: FileEntry[];
  tree: Record<string, unknown>;
}

export interface SourceChunk {
  chunk_id: string;
  file_path: string;
  language: string;
  symbol_name: string | null;
  symbol_type: string;
  start_line: number;
  end_line: number;
  content: string;
  context_header: string;
  imports?: string[];
  semantic_score?: number;
  lexical_score?: number;
  rrf_score?: number;
}

export interface TraceStep {
  step: number;
  stage: string; // analyze | retrieve | grade | hop | answer | ...
  description: string;
  [key: string]: unknown;
}

export interface QueryResponse {
  answer: string;
  agent_trace: TraceStep[];
  sources: SourceChunk[];
  hops_used: number;
}

export interface IndexEvent {
  stage: string;
  progress: number;
  message: string;
  repo_id?: string;
  cached?: boolean;
  total_files?: number;
  total_chunks?: number;
  fetched_files?: number;
  batch?: number;
  total_batches?: number;
  [key: string]: unknown;
}

export interface ChatTurn {
  id: string;
  role: "user" | "assistant";
  content: string;
  trace?: TraceStep[];
  sources?: SourceChunk[];
  hopsUsed?: number;
  pending?: boolean;
  error?: string;
}

export interface CitationRef {
  filePath: string;
  startLine: number;
  endLine: number;
}
