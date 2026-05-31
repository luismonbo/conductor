interface UserMessageProps {
  text: string;
}

export function UserMessage({ text }: UserMessageProps) {
  return (
    <div style={{ display: 'flex', justifyContent: 'flex-end', padding: '4px 16px' }}>
      <div
        style={{
          maxWidth: '70%',
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: '12px 12px 2px 12px',
          padding: '10px 14px',
          fontFamily: 'var(--mono)',
          fontSize: 'var(--text-sm)',
          color: 'var(--text)',
          lineHeight: 1.5,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {text}
      </div>
    </div>
  );
}
