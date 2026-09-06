interface ToolCallPillProps {
  name: string;
  args: Record<string, unknown>;
}

export function ToolCallPill({ name, args }: ToolCallPillProps) {
  return (
    <details className="my-1 border-l-2 border-accent pl-2.5">
      <summary className="flex list-none select-none items-center gap-1.5 py-0.5 font-mono text-xs text-fg-muted">
        <span>tool_call</span>
        <span className="font-medium text-fg">{name}</span>
      </summary>
      <pre className="mt-1.5 overflow-x-auto whitespace-pre-wrap break-all rounded-sm border border-border bg-surface-raised p-2 font-mono text-xs text-fg-code">
        {JSON.stringify(args, null, 2)}
      </pre>
    </details>
  );
}
