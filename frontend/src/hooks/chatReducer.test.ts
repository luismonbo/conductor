import { describe, it, expect } from 'vitest';
import { chatReducer, initialState } from '@/hooks/useChatStream';
import type { ChatState } from '@/hooks/useChatStream';

function withStreamingAssistant(): ChatState {
  return chatReducer(initialState, {
    type: 'SEND_USER_MESSAGE',
    text: 'hello',
    assistantId: 'assist-1',
  });
}

describe('chatReducer — APPEND_THINKING (token streaming path)', () => {
  it('creates a thinking block on the first token', () => {
    const state = withStreamingAssistant();
    const next = chatReducer(state, { type: 'APPEND_THINKING', text: 'Hello' });

    const assistant = next.messages.find((m) => m.role === 'assistant');
    expect(assistant?.blocks).toHaveLength(1);
    expect(assistant?.blocks[0]).toEqual({ kind: 'thinking', text: 'Hello' });
  });

  it('appends a second token to the same thinking block', () => {
    let state = withStreamingAssistant();
    state = chatReducer(state, { type: 'APPEND_THINKING', text: 'Hello' });
    state = chatReducer(state, { type: 'APPEND_THINKING', text: ' world' });

    const assistant = state.messages.find((m) => m.role === 'assistant');
    expect(assistant?.blocks).toHaveLength(1);
    expect(assistant?.blocks[0]).toEqual({ kind: 'thinking', text: 'Hello world' });
  });

  it('accumulates many tokens into one thinking block', () => {
    let state = withStreamingAssistant();
    for (const word of ['The', ' answer', ' is', ' 42']) {
      state = chatReducer(state, { type: 'APPEND_THINKING', text: word });
    }

    const assistant = state.messages.find((m) => m.role === 'assistant');
    expect(assistant?.blocks).toHaveLength(1);
    expect(assistant?.blocks[0].text).toBe('The answer is 42');
  });

  it('does not mutate the previous state', () => {
    const state = withStreamingAssistant();
    const next = chatReducer(state, { type: 'APPEND_THINKING', text: 'Hi' });
    expect(state.messages).not.toBe(next.messages);
  });
});