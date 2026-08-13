interface ModelPickerProps {
  models: string[];
  value: string;
  onChange: (model: string) => void;
  disabled: boolean;
}

export function ModelPicker({ models, value, onChange, disabled }: ModelPickerProps) {
  if (models.length === 0) return null;
  return (
    <select
      aria-label="model"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      style={{
        fontFamily: 'var(--mono)',
        fontSize: 'var(--text-xs)',
        color: 'var(--text-muted)',
        background: 'transparent',
        border: '1px solid var(--border)',
        borderRadius: '4px',
        padding: '4px 8px',
        letterSpacing: '0.06em',
        cursor: disabled ? 'default' : 'pointer',
      }}
    >
      {models.map((m) => (
        <option key={m} value={m}>{m}</option>
      ))}
    </select>
  );
}
