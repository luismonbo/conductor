import { describe, it, expect, vi, afterEach } from 'vitest';
import { HttpError, describeHttpError, streamChat } from '@/api';

afterEach(() => vi.unstubAllGlobals());

describe('describeHttpError', () => {
  it.each([
    [429, "You're sending messages too quickly. Please wait a minute and try again."],
    [413, 'That message is too long. Please shorten it.'],
    [404, 'That conversation has expired. Your next message will start a new one.'],
    [500, 'The assistant is unavailable right now. Please try again later.'],
    [401, 'The assistant is unavailable right now. Please try again later.'],
  ])('maps %i to a sentence', (status, sentence) => {
    expect(describeHttpError(status)).toBe(sentence);
  });
});

describe('streamChat', () => {
  it('throws an HttpError carrying the status and the sentence, never the status line', async () => {
    vi.stubGlobal('fetch', async () => ({ ok: false, status: 429, statusText: 'Too Many Requests' }));

    const gen = streamChat({ message: 'hi' }, new AbortController().signal);
    const err = await gen.next().then(() => null, (e: unknown) => e);

    expect(err).toBeInstanceOf(HttpError);
    expect((err as HttpError).status).toBe(429);
    expect((err as HttpError).message).toBe(describeHttpError(429));
    expect((err as HttpError).message).not.toContain('Too Many Requests');
  });
});
