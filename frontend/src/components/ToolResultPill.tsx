interface ToolResultPillProps {
  text: string;
  name: string;
  is_error: boolean;
}

export function ToolResultPill({ text, name, is_error }: ToolResultPillProps) {
  return (
    <details
      className={`my-1 pl-2.5 ${is_error ? 'error-result border-l-2 border-error' : 'border-l-2 border-border'}`}
    >
      <summary className="flex list-none select-none items-center gap-1.5 py-0.5 font-mono text-xs text-fg-muted">
        <span>tool_result</span>
        <span className={is_error ? 'text-error' : 'text-fg'}>{name}</span>
        {is_error && <span className="font-mono text-xs text-error">error</span>}
      </summary>
      <pre
        className={`mt-1.5 overflow-x-auto whitespace-pre-wrap break-all rounded-sm border bg-surface-raised p-2 font-mono text-xs ${
          is_error ? 'border-error/30 text-error' : 'border-border text-fg'
        }`}
      >
        {text}
      </pre>
    </details>
  );
}
