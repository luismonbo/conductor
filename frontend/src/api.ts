import { parseSSEStream } from '@/lib/sseParser';
import {
  isAgentEvent,
  isThreadIdPayload,
  type AgentEvent,
  type CancelResponse,
  type ChatRequest,
  type HealthResponse,
  type ModelsResponse,
  type ResumeRequest,
  type ThreadIdPayload,
  type ThreadMessagesResponse,
  type ThreadSummary,
} from '@/types';

const BASE = import.meta.env.VITE_API_URL ?? '/api';

// What a person reads when a request fails: fixed sentences, never a raw status
// line. The site's chat widget (luismonbo.com) uses the same wording.
export const NETWORK_ERROR_TEXT =
  "Couldn't reach the assistant. Check your connection and try again.";
export const UNAVAILABLE_TEXT = 'The assistant is unavailable right now. Please try again later.';

export function describeHttpError(status: number): string {
  switch (status) {
    case 429:
      return "You're sending messages too quickly. Please wait a minute and try again.";
    case 413:
      return 'That message is too long. Please shorten it.';
    case 404:
      return 'That conversation has expired. Your next message will start a new one.';
    default:
      return UNAVAILABLE_TEXT;
  }
}

/** A non-OK response: `message` is the sentence to show, `status` decides what else happens. */
export class HttpError extends Error {
  readonly status: number;

  constructor(status: number) {
    super(describeHttpError(status));
    this.name = 'HttpError';
    this.status = status;
  }
}

export async function* streamChat(
  payload: ChatRequest,
  signal: AbortSignal,
): AsyncGenerator<ThreadIdPayload | AgentEvent> {
  const response = await fetch(`${BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    throw new HttpError(response.status);
  }

  if (!response.body) {
    throw new Error('Response body is null');
  }

  for await (const raw of parseSSEStream(response.body)) {
    if (isThreadIdPayload(raw) || isAgentEvent(raw)) {
      yield raw;
    }
  }
}

export async function* resumeChat(
  threadId: string,
  decision: Record<string, unknown>,
  signal: AbortSignal,
): AsyncGenerator<ThreadIdPayload | AgentEvent> {
  const payload: ResumeRequest = { decision };
  const response = await fetch(`${BASE}/resume/${threadId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    throw new HttpError(response.status);
  }

  if (!response.body) {
    throw new Error('Response body is null');
  }

  for await (const raw of parseSSEStream(response.body)) {
    if (isThreadIdPayload(raw) || isAgentEvent(raw)) {
      yield raw;
    }
  }
}

export async function cancelChat(threadId: string): Promise<CancelResponse> {
  const response = await fetch(`${BASE}/cancel/${threadId}`, { method: 'POST' });
  if (!response.ok) {
    throw new HttpError(response.status);
  }
  return response.json() as Promise<CancelResponse>;
}

export async function fetchModels(): Promise<ModelsResponse> {
  const response = await fetch(`${BASE}/models`);
  if (!response.ok) {
    throw new HttpError(response.status);
  }
  return response.json() as Promise<ModelsResponse>;
}

export async function fetchThreads(): Promise<{ threads: ThreadSummary[] }> {
  const response = await fetch(`${BASE}/threads`);
  if (!response.ok) {
    throw new HttpError(response.status);
  }
  return response.json() as Promise<{ threads: ThreadSummary[] }>;
}

export async function fetchThreadMessages(threadId: string): Promise<ThreadMessagesResponse> {
  const response = await fetch(`${BASE}/threads/${threadId}`);
  if (!response.ok) {
    throw new HttpError(response.status);
  }
  return response.json() as Promise<ThreadMessagesResponse>;
}

export async function healthCheck(): Promise<HealthResponse> {
  const response = await fetch(`${BASE}/health`);
  if (!response.ok) {
    throw new HttpError(response.status);
  }
  return response.json() as Promise<HealthResponse>;
}
