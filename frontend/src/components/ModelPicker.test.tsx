import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ModelPicker } from './ModelPicker';

describe('ModelPicker', () => {
  it('renders nothing when no models', () => {
    const { container } = render(
      <ModelPicker models={[]} value="" onChange={() => {}} disabled={false} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('lists models and fires onChange', async () => {
    const onChange = vi.fn();
    render(
      <ModelPicker
        models={['claude', 'local-gemma']}
        value="local-gemma"
        onChange={onChange}
        disabled={false}
      />,
    );
    const select = screen.getByLabelText('model');
    expect(select).toHaveValue('local-gemma');
    await userEvent.selectOptions(select, 'claude');
    expect(onChange).toHaveBeenCalledWith('claude');
  });

  it('disables while streaming', () => {
    render(
      <ModelPicker models={['claude']} value="claude" onChange={() => {}} disabled />,
    );
    expect(screen.getByLabelText('model')).toBeDisabled();
  });
});
