import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useChatStream } from '@/hooks/useChatStream';

afterEach(() => vi.unstubAllGlobals());

/** A fetch response whose body streams these SSE frames. */
function sseResponse(frames: readonly unknown[]) {
  const text = frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join('');
  return {
    ok: true,
    status: 200,
    body: new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(text));
        controller.close();
      },
    }),
  };
}

const proposalFrame = {
  type: 'interrupt', text: '', name: '', call_id: '', is_error: false, stopped_reason: '',
  args: { mode: 'memory_proposal', proposed: 'likes tea', call_id: 'c1', interrupt_id: 'i-1' },
};

describe('useChatStream', () => {
  it("answering a card names that card's interrupt", async () => {
    // The server refuses an answer for any other interrupt, so a card answered twice,
    // or in another tab, cannot approve the card after it.
    const sentBodies: unknown[] = [];
    vi.stubGlobal('fetch', async (_url: string, init?: RequestInit) => {
      sentBodies.push(JSON.parse(String(init?.body)));
      return sseResponse([{ thread_id: 't-1' }, proposalFrame]);
    });
    const { result } = renderHook(() => useChatStream());

    await act(() => result.current.sendMessage('remember I like tea'));
    await act(() => result.current.resumeStream({ action: 'approve' }));

    expect(sentBodies[1]).toEqual({ interrupt_id: 'i-1', decision: { action: 'approve' } });
  });
});
