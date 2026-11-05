import { useMemo, useState } from "react";
import {
  Braces,
  ChevronDown,
  ChevronRight,
  File,
  FileCode2,
  FileText,
  Folder,
  FolderOpen,
  Hash,
} from "lucide-react";

import type { FileEntry, SourceChunk } from "../types";

interface TreeNode {
  name: string;
  path: string;
  children: Map<string, TreeNode>;
  file?: FileEntry;
}

function buildTree(files: FileEntry[]): TreeNode {
  const root: TreeNode = { name: "", path: "", children: new Map() };
  for (const file of files) {
    const parts = file.file_path.split("/");
    let node = root;
    parts.forEach((part, i) => {
      if (!node.children.has(part)) {
        node.children.set(part, {
          name: part,
          path: parts.slice(0, i + 1).join("/"),
          children: new Map(),
        });
      }
      node = node.children.get(part)!;
      if (i === parts.length - 1) node.file = file;
    });
  }
  return root;
}

function languageIcon(language: string) {
  if (language === "markdown" || language === "rst") {
    return <FileText className="w-3.5 h-3.5 text-slate-500 shrink-0" />;
  }
  return <FileCode2 className="w-3.5 h-3.5 text-slate-500 shrink-0" />;
}

interface NodeProps {
  node: TreeNode;
  depth: number;
  selectedPath: string | null;
  onSelectFile: (file: FileEntry) => void;
}

function TreeNodeView({ node, depth, selectedPath, onSelectFile }: NodeProps) {
  const [open, setOpen] = useState(depth < 1);
  const isFolder = !node.file && node.children.size > 0;
  const sorted = [...node.children.values()].sort((a, b) =>
    a.name.localeCompare(b.name),
  );
  if (node.file) {
    const selected = selectedPath === node.file.file_path;
    return (
      <button
        type="button"
        onClick={() => onSelectFile(node.file!)}
        className={`w-full flex items-center gap-1.5 py-1 pr-2 rounded text-left text-[13px] ${
          selected
            ? "bg-lens/10 text-lens-soft"
            : "text-slate-300 hover:bg-ink-800"
        }`}
        style={{ paddingLeft: `${depth * 14 + 8}px` }}
        title={node.file.file_path}
      >
        {languageIcon(node.file.language)}
        <span className="truncate flex-1">{node.name}</span>
        <span
          className={`font-mono text-[10px] px-1.5 py-px rounded-full border ${
            selected
              ? "border-lens/40 text-lens"
              : "border-ink-600 text-slate-500"
          }`}
          title={`${node.file.chunks} AST chunks in this file`}
        >
          {node.file.chunks}
        </span>
      </button>
    );
  }
  if (!isFolder) return null;
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-1 py-1 pr-2 rounded text-left text-[13px] text-slate-400 hover:bg-ink-800"
        style={{ paddingLeft: `${depth * 14 + 4}px` }}
      >
        {open ? (
          <ChevronDown className="w-3.5 h-3.5 shrink-0" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 shrink-0" />
        )}
        {open ? (
          <FolderOpen className="w-3.5 h-3.5 text-lens/70 shrink-0" />
        ) : (
          <Folder className="w-3.5 h-3.5 text-lens/70 shrink-0" />
        )}
        <span className="truncate">{node.name}</span>
      </button>
      {open &&
        sorted.map((child) => (
          <TreeNodeView
            key={child.path}
            node={child}
            depth={depth + 1}
            selectedPath={selectedPath}
            onSelectFile={onSelectFile}
          />
        ))}
    </div>
  );
}

const SYMBOL_ICON: Record<string, string> = {
  function: "ƒ",
  method: "ƒ",
  class: "C",
  module_header: "M",
  doc: "§",
};

interface SidebarProps {
  files: FileEntry[];
  totalFiles: number;
  totalChunks: number;
  selectedFile: FileEntry | null;
  fileChunks: SourceChunk[] | null;
  fileChunksLoading: boolean;
  onSelectFile: (file: FileEntry) => void;
  onInspectChunk: (chunk: SourceChunk) => void;
}

export default function Sidebar({
  files,
  totalFiles,
  totalChunks,
  selectedFile,
  fileChunks,
  fileChunksLoading,
  onSelectFile,
  onInspectChunk,
}: SidebarProps) {
  const tree = useMemo(() => buildTree(files), [files]);
  const rootChildren = [...tree.children.values()].sort((a, b) =>
    a.name.localeCompare(b.name),
  );

  return (
    <aside className="w-72 shrink-0 hidden md:flex flex-col border-r border-ink-700 bg-ink-900 min-h-0">
      <div className="px-3.5 pt-3.5 pb-2 flex items-center gap-2">
        <Hash className="w-4 h-4 text-lens" />
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          File &amp; AST Explorer
        </h2>
        <span className="ml-auto font-mono text-[11px] text-slate-500">
          {totalFiles} files · {totalChunks} chunks
        </span>
      </div>
      <div className="flex-1 overflow-y-auto px-1.5 pb-2 min-h-0">
        {files.length === 0 && (
          <p className="text-sm text-slate-500 px-2.5 py-3 leading-relaxed">
            No repository indexed yet. Index one to explore its AST here.
          </p>
        )}
        {rootChildren.map((child) => (
          <TreeNodeView
            key={child.path}
            node={child}
            depth={0}
            selectedPath={selectedFile?.file_path ?? null}
            onSelectFile={onSelectFile}
          />
        ))}
      </div>

      {selectedFile && (
        <div className="border-t border-ink-700 max-h-[46%] flex flex-col min-h-0">
          <div className="px-3.5 py-2.5">
            <div className="font-mono text-xs text-lens-soft truncate" title={selectedFile.file_path}>
              {selectedFile.file_path}
            </div>
            <div className="text-[11px] text-slate-500 mt-0.5 flex items-center gap-1.5">
              <Braces className="w-3 h-3" />
              {selectedFile.language} · {selectedFile.chunks} AST chunks — click to inspect
            </div>
          </div>
          <div className="overflow-y-auto px-1.5 pb-2 min-h-0">
            {fileChunksLoading && (
              <p className="text-xs text-slate-500 px-2 py-1">Loading chunks…</p>
            )}
            {!fileChunksLoading &&
              (fileChunks ?? []).map((chunk) => (
                <button
                  key={chunk.chunk_id}
                  type="button"
                  onClick={() => onInspectChunk(chunk)}
                  className="w-full flex items-center gap-2 px-2 py-1.5 rounded text-left hover:bg-ink-800 group"
                >
                  <span className="w-5 h-5 shrink-0 rounded bg-ink-700 text-lens font-mono text-[11px] flex items-center justify-center">
                    {SYMBOL_ICON[chunk.symbol_type] ?? "·"}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block font-mono text-xs text-slate-200 truncate group-hover:text-lens-soft">
                      {chunk.symbol_name ?? "(module)"}
                    </span>
                    <span className="block text-[10px] text-slate-500 font-mono">
                      {chunk.symbol_type} · L{chunk.start_line}-{chunk.end_line}
                    </span>
                  </span>
                </button>
              ))}
            {!fileChunksLoading && fileChunks && fileChunks.length === 0 && (
              <p className="text-xs text-slate-500 px-2 py-1 flex items-center gap-1.5">
                <File className="w-3 h-3" /> No chunks stored for this file.
              </p>
            )}
          </div>
        </div>
      )}
    </aside>
  );
}
