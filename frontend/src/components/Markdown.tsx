import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { FileCode2 } from "lucide-react";

import type { CitationRef } from "../types";

// Citations in agent answers look like [src/app/main.py:42-88]. They
// are rewritten to a special #cite/ link so ReactMarkdown renders them
// through the `a` component override below as clickable badges.
const CITATION_RE = /\[([^\]\s:]+):(\d+)-(\d+)\]/g;
const CITE_HREF_RE = /^#cite\/(.+)\/(\d+)\/(\d+)$/;

export function linkifyCitations(markdown: string): string {
  return markdown.replace(CITATION_RE, (_m, file: string, s: string, e: string) => {
    const label = `${file}:${s}-${e}`;
    return `[${label}](#cite/${encodeURIComponent(file)}/${s}/${e})`;
  });
}

interface MarkdownProps {
  text: string;
  onCitation: (ref: CitationRef) => void;
}

export default function Markdown({ text, onCitation }: MarkdownProps) {
  const components: Components = {
    a({ href, children }) {
      const match = href ? CITE_HREF_RE.exec(href) : null;
      if (match) {
        const filePath = decodeURIComponent(match[1]);
        const startLine = Number(match[2]);
        const endLine = Number(match[3]);
        return (
          <button
            type="button"
            onClick={() => onCitation({ filePath, startLine, endLine })}
            className="inline-flex items-center gap-1 align-baseline font-mono text-[0.78em] font-medium text-lens-soft bg-lens/10 hover:bg-lens/20 border border-lens/30 rounded px-1.5 py-px mx-0.5 transition-colors"
            title="Open in Code Inspector"
          >
            <FileCode2 className="w-3 h-3" />
            {filePath}:{startLine}-{endLine}
          </button>
        );
      }
      return (
        <a
          href={href}
          target="_blank"
          rel="noreferrer"
          className="text-sky-300 underline decoration-sky-300/40 hover:decoration-sky-300"
        >
          {children}
        </a>
      );
    },
  };
  return (
    <div className="md-body">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {linkifyCitations(text)}
      </ReactMarkdown>
    </div>
  );
}
