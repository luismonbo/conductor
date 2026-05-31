interface ToolResultPillProps {
  text: string;
  name: string;
  is_error: boolean;
}

export function ToolResultPill({ text, name, is_error }: ToolResultPillProps) {
  return (
    <details
      style={{
        borderLeft: `2px solid ${is_error ? '#f87171' : 'var(--border)'}`,
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
        <span style={{ color: is_error ? '#f87171' : 'var(--text-muted)', fontSize: '8px' }}>▶</span>
        <span>tool_result</span>
        <span style={{ color: is_error ? '#f87171' : 'var(--text)' }}>{name}</span>
        {is_error && (
          <span style={{ color: '#f87171', fontSize: 'var(--text-xs)' }}>error</span>
        )}
      </summary>
      <pre
        style={{
          margin: '6px 0 0',
          padding: '8px',
          background: 'var(--surface-raised)',
          border: `1px solid ${is_error ? 'rgba(248,113,113,0.3)' : 'var(--border)'}`,
          borderRadius: '4px',
          fontFamily: 'var(--mono)',
          fontSize: 'var(--text-xs)',
          color: is_error ? '#f87171' : 'var(--text)',
          overflowX: 'auto',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-all',
        }}
      >
        {text}
      </pre>
    </details>
  );
}
