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
      className="rounded-sm border border-border bg-transparent px-2 py-1 font-mono text-xs tracking-wide text-fg-muted disabled:cursor-default enabled:cursor-pointer"
    >
      {models.map((m) => (
        <option key={m} value={m}>{m}</option>
      ))}
    </select>
  );
}
