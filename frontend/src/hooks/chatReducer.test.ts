import { describe, it, expect } from 'vitest';
import { chatReducer, initialState, threadMessagesToConversation } from '@/hooks/useChatStream';
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
    expect(assistant?.blocks[0]).toEqual({ kind: 'thinking', text: 'The answer is 42' });
  });

  it('does not mutate the previous state', () => {
    const state = withStreamingAssistant();
    const next = chatReducer(state, { type: 'APPEND_THINKING', text: 'Hi' });
    expect(state.messages).not.toBe(next.messages);
  });
});

describe('LOAD_THREAD / NEW_THREAD', () => {
  it('LOAD_THREAD replaces messages and thread id', () => {
    const loaded = [{ id: 'x', role: 'user' as const, text: 'old question' }];
    const state = chatReducer(
      { ...initialState, threadId: 'other', messages: [] },
      { type: 'LOAD_THREAD', threadId: 't-42', messages: loaded },
    );
    expect(state.threadId).toBe('t-42');
    expect(state.messages).toEqual(loaded);
    expect(state.streamStatus).toBe('idle');
  });

  it('NEW_THREAD clears the conversation', () => {
    const state = chatReducer(
      { ...initialState, threadId: 't-42', messages: [{ id: 'x', role: 'user', text: 'hi' }] },
      { type: 'NEW_THREAD' },
    );
    expect(state.threadId).toBeNull();
    expect(state.messages).toEqual([]);
  });
});

describe('threadMessagesToConversation', () => {
  it('maps user/assistant/tool into conversation messages', () => {
    const out = threadMessagesToConversation([
      { role: 'user', content: 'add 2+2', tool_calls: [], tool_call_id: '', name: '' },
      {
        role: 'assistant', content: '',
        tool_calls: [{ name: 'calculator', args: { a: 2 }, call_id: 'c1' }],
        tool_call_id: '', name: '',
      },
      { role: 'tool', content: '4', tool_calls: [], tool_call_id: 'c1', name: 'calculator' },
      { role: 'assistant', content: 'The answer is 4.', tool_calls: [], tool_call_id: '', name: '' },
    ]);
    expect(out).toHaveLength(3); // user, assistant(+tool blocks), assistant
    const first = out[1];
    expect(first.role).toBe('assistant');
    if (first.role === 'assistant') {
      expect(first.blocks.map((b) => b.kind)).toEqual(['tool_call', 'tool_result']);
    }
    const last = out[2];
    if (last.role === 'assistant') expect(last.finalText).toBe('The answer is 4.');
  });
});