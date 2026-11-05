import { highlightCode } from "../lib/highlight";

interface CodeViewProps {
  code: string;
  startLine: number;
}

/** Code block with a line-number gutter offset to the chunk's real
 * line numbers in the source file. */
export default function CodeView({ code, startLine }: CodeViewProps) {
  const lines = code.replace(/\n$/, "").split("\n");
  return (
    <div className="overflow-auto bg-ink-900 rounded-lg border border-ink-700">
      <pre className="font-mono text-[12.5px] leading-[1.55] py-3 min-w-max">
        {lines.map((line, i) => (
          <div key={i} className="flex px-0 hover:bg-ink-800/50">
            <span className="w-12 shrink-0 text-right pr-4 select-none text-slate-600">
              {startLine + i}
            </span>
            <code className="text-slate-200 whitespace-pre pr-4">
              {highlightCode(line)}
            </code>
          </div>
        ))}
      </pre>
    </div>
  );
}
