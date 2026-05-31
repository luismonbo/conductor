interface ToolCallPillProps {
  name: string;
  args: Record<string, unknown>;
}

export function ToolCallPill({ name, args }: ToolCallPillProps) {
  return (
    <details
      style={{
        borderLeft: '2px solid var(--accent)',
        paddingLeft: '10px',
        margin: '4px 0',
      }}
    >
      <summary
        style={{
          cursor: 'pointer',
          fontFamily: 'var(--mono)',
          fontSize: 'var(--text-xs)',
          color: 'var(--text-muted)',
          listStyle: 'none',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '3px 0',
          userSelect: 'none',
        }}
      >
        <span style={{ color: 'var(--accent)', fontSize: '8px' }}>▶</span>
        <span>tool_call</span>
        <span style={{ color: 'var(--text)', fontWeight: 500 }}>{name}</span>
      </summary>
      <pre
        style={{
          margin: '6px 0 0',
          padding: '8px',
          background: 'var(--surface-raised)',
          border: '1px solid var(--border)',
          borderRadius: '4px',
          fontFamily: 'var(--mono)',
          fontSize: 'var(--text-xs)',
          color: 'var(--text-code)',
          overflowX: 'auto',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-all',
        }}
      >
        {JSON.stringify(args, null, 2)}
      </pre>
    </details>
  );
}
