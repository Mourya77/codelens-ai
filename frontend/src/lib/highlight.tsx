// A small, dependency-free syntax highlighter.
//
// It tokenizes with a single combined regex (comments, strings,
// numbers, keywords, types, function calls) and renders React spans —
// no dangerouslySetInnerHTML, no external highlighter bundle. It is
// intentionally generic across Python / TS / JS / Go / Java / C++,
// which covers the languages the chunker indexes.

import type { ReactNode } from "react";

const KEYWORDS = new Set([
  // shared
  "if", "else", "elif", "for", "while", "return", "break", "continue",
  "class", "def", "function", "import", "from", "as", "with", "try",
  "except", "finally", "raise", "throw", "throws", "new", "async",
  "await", "yield", "lambda", "pass", "in", "is", "not", "and", "or",
  "const", "let", "var", "export", "default", "extends", "implements",
  "interface", "type", "enum", "struct", "package", "func", "go",
  "public", "private", "protected", "static", "void", "int", "str",
  "bool", "float", "string", "boolean", "number", "this", "self",
  "super", "null", "none", "true", "false", "True", "False", "None",
  "switch", "case", "match", "namespace", "using", "template",
]);

const TOKEN_RE =
  /(\/\/[^\n]*|#[^\n]*|\/\*[\s\S]*?\*\/)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)|\b(\d+(?:\.\d+)?)\b|([A-Za-z_][A-Za-z0-9_]*)/g;

export function highlightCode(code: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let key = 0;
  let match: RegExpExecArray | null;
  TOKEN_RE.lastIndex = 0;
  while ((match = TOKEN_RE.exec(code)) !== null) {
    if (match.index > last) {
      out.push(code.slice(last, match.index));
    }
    const [full, comment, str, num, ident] = match;
    if (comment) {
      out.push(
        <span key={key++} className="text-slate-500 italic">
          {full}
        </span>,
      );
    } else if (str) {
      out.push(
        <span key={key++} className="text-amber-300/90">
          {full}
        </span>,
      );
    } else if (num) {
      out.push(
        <span key={key++} className="text-orange-300">
          {full}
        </span>,
      );
    } else if (ident) {
      const after = code.slice(TOKEN_RE.lastIndex).trimStart();
      if (KEYWORDS.has(ident)) {
        out.push(
          <span key={key++} className="text-fuchsia-300/90">
            {full}
          </span>,
        );
      } else if (after.startsWith("(")) {
        out.push(
          <span key={key++} className="text-sky-300">
            {full}
          </span>,
        );
      } else if (/^[A-Z]/.test(ident)) {
        out.push(
          <span key={key++} className="text-teal-200">
            {full}
          </span>,
        );
      } else {
        out.push(full);
      }
    }
    last = TOKEN_RE.lastIndex;
  }
  if (last < code.length) out.push(code.slice(last));
  return out;
}
