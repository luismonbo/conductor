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
    throw new Error(`Stream request failed: ${response.status} ${response.statusText}`);
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
    throw new Error(`Resume request failed: ${response.status} ${response.statusText}`);
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
    throw new Error(`Cancel request failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<CancelResponse>;
}

export async function fetchModels(): Promise<ModelsResponse> {
  const response = await fetch(`${BASE}/models`);
  if (!response.ok) {
    throw new Error(`Models request failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<ModelsResponse>;
}

export async function fetchThreads(): Promise<{ threads: ThreadSummary[] }> {
  const response = await fetch(`${BASE}/threads`);
  if (!response.ok) {
    throw new Error(`Threads request failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<{ threads: ThreadSummary[] }>;
}

export async function fetchThreadMessages(threadId: string): Promise<ThreadMessagesResponse> {
  const response = await fetch(`${BASE}/threads/${threadId}`);
  if (!response.ok) {
    throw new Error(`Thread request failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<ThreadMessagesResponse>;
}

export async function healthCheck(): Promise<HealthResponse> {
  const response = await fetch(`${BASE}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<HealthResponse>;
}
