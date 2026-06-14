import { useCallback, useEffect, useReducer, useRef } from 'react';
import { streamChat, cancelChat, resumeChat } from '@/api';
import {
  isThreadIdPayload,
  type AgentEvent,
  type AssistantMessage,
  type ConversationMessage,
  type InterruptPayload,
  type MessageBlock,
  type StreamStatus,
  type ThinkingBlock,
  type ThreadIdPayload,
} from '@/types';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

export interface ChatState {
  threadId: string | null;
  messages: ConversationMessage[];
  streamStatus: StreamStatus;
  currentTool: string | null;
  inputValue: string;
  errorMessage: string | null;
  interruptPayload: InterruptPayload | null;
}

export const initialState: ChatState = {
  threadId: null,
  messages: [],
  streamStatus: 'idle',
  currentTool: null,
  inputValue: '',
  errorMessage: null,
  interruptPayload: null,
};

// ---------------------------------------------------------------------------
// Action union
// ---------------------------------------------------------------------------

type ChatAction =
  | { type: 'SET_INPUT'; value: string }
  | { type: 'SEND_USER_MESSAGE'; text: string; assistantId: string }
  | { type: 'SET_THREAD_ID'; id: string }
  | { type: 'APPEND_THINKING'; text: string }
  | { type: 'ADD_TOOL_CALL'; name: string; args: Record<string, unknown>; call_id: string }
  | { type: 'ADD_TOOL_RESULT'; text: string; name: string; call_id: string; is_error: boolean }
  | { type: 'STREAM_FINAL'; text: string; stopped_reason: string }
  | { type: 'STREAM_ERROR'; text: string }
  | { type: 'STREAM_INTERRUPT'; payload: InterruptPayload }
  | { type: 'STREAM_CANCELLED' };

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getLastAssistantMessage(messages: ConversationMessage[]): AssistantMessage | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === 'assistant' && msg.isStreaming) return msg;
  }
  return null;
}

function updateLastStreamingMessage(
  messages: ConversationMessage[],
  updater: (msg: AssistantMessage) => AssistantMessage,
): ConversationMessage[] {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === 'assistant' && msg.isStreaming) {
      return [...messages.slice(0, i), updater(msg), ...messages.slice(i + 1)];
    }
  }
  return messages;
}

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case 'SET_INPUT':
      return { ...state, inputValue: action.value };

    case 'SEND_USER_MESSAGE': {
      const clearedMessages = state.messages.map((msg) =>
        msg.role === 'assistant' && msg.interruptPayload
          ? { ...msg, interruptPayload: undefined }
          : msg
      );
      const assistantMsg: AssistantMessage = {
        id: action.assistantId,
        role: 'assistant',
        blocks: [],
        isStreaming: true,
      };
      const newMessages = action.text.trim()
        ? [
            ...clearedMessages,
            { id: crypto.randomUUID(), role: 'user' as const, text: action.text },
            assistantMsg,
          ]
        : [...clearedMessages, assistantMsg];
      return {
        ...state,
        messages: newMessages,
        streamStatus: 'streaming',
        inputValue: '',
        currentTool: null,
        errorMessage: null,
        interruptPayload: null,
      };
    }

    case 'SET_THREAD_ID':
      return { ...state, threadId: action.id };

    case 'APPEND_THINKING': {
      const messages = updateLastStreamingMessage(state.messages, (msg) => {
        const lastBlock = msg.blocks[msg.blocks.length - 1];
        let newBlocks: MessageBlock[];
        if (lastBlock && lastBlock.kind === 'thinking') {
          const updatedBlock: ThinkingBlock = { ...lastBlock, text: lastBlock.text + action.text };
          newBlocks = [...msg.blocks.slice(0, -1), updatedBlock];
        } else {
          newBlocks = [...msg.blocks, { kind: 'thinking' as const, text: action.text }];
        }
        return { ...msg, blocks: newBlocks };
      });
      return { ...state, messages };
    }

    case 'ADD_TOOL_CALL': {
      const messages = updateLastStreamingMessage(state.messages, (msg) => ({
        ...msg,
        blocks: [
          ...msg.blocks,
          { kind: 'tool_call' as const, name: action.name, args: action.args, call_id: action.call_id },
        ],
      }));
      return { ...state, messages, currentTool: action.name };
    }

    case 'ADD_TOOL_RESULT': {
      const messages = updateLastStreamingMessage(state.messages, (msg) => ({
        ...msg,
        blocks: [
          ...msg.blocks,
          {
            kind: 'tool_result' as const,
            text: action.text,
            name: action.name,
            call_id: action.call_id,
            is_error: action.is_error,
          },
        ],
      }));
      return { ...state, messages, currentTool: null };
    }

    case 'STREAM_FINAL': {
      const lastMsg = getLastAssistantMessage(state.messages);
      const lastThinkingBlock = lastMsg?.blocks
        .filter((b): b is ThinkingBlock => b.kind === 'thinking')
        .at(-1);
      const shouldSetFinalText = action.text !== lastThinkingBlock?.text;
      const messages = updateLastStreamingMessage(state.messages, (msg) => ({
        ...msg,
        isStreaming: false,
        ...(shouldSetFinalText ? { finalText: action.text } : {}),
      }));
      return { ...state, messages, streamStatus: 'done', currentTool: null, errorMessage: null };
    }

    case 'STREAM_ERROR': {
      if (state.streamStatus === 'done') return state;
      const messages = updateLastStreamingMessage(state.messages, (msg) => ({
        ...msg,
        isStreaming: false,
      }));
      return { ...state, messages, streamStatus: 'error', currentTool: null, errorMessage: action.text };
    }

    case 'STREAM_INTERRUPT': {
      const messages = updateLastStreamingMessage(state.messages, (msg) => ({
        ...msg,
        isStreaming: false,
        interruptPayload: action.payload,
      }));
      return {
        ...state,
        messages,
        streamStatus: 'interrupted',
        currentTool: null,
        interruptPayload: action.payload,
      };
    }

    case 'STREAM_CANCELLED': {
      const messages = updateLastStreamingMessage(state.messages, (msg) => ({
        ...msg,
        isStreaming: false,
      }));
      return { ...state, messages, streamStatus: 'idle', currentTool: null, errorMessage: null };
    }

    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useChatStream() {
  const [state, dispatch] = useReducer(chatReducer, initialState);
  const abortRef = useRef<AbortController | null>(null);

  const _consumeStream = useCallback(
    async (gen: AsyncGenerator<ThreadIdPayload | AgentEvent>) => {
      try {
        for await (const event of gen) {
          if (isThreadIdPayload(event)) {
            dispatch({ type: 'SET_THREAD_ID', id: event.thread_id });
            continue;
          }

          // event is AgentEvent from here
          switch (event.type) {
            case 'token':
              dispatch({ type: 'APPEND_THINKING', text: event.text });
              break;
            case 'thinking':
              dispatch({ type: 'APPEND_THINKING', text: event.text });
              break;
            case 'tool_call':
              dispatch({
                type: 'ADD_TOOL_CALL',
                name: event.name,
                args: event.args,
                call_id: event.call_id,
              });
              break;
            case 'tool_result':
              dispatch({
                type: 'ADD_TOOL_RESULT',
                text: event.text,
                name: event.name,
                call_id: event.call_id,
                is_error: event.is_error,
              });
              break;
            case 'final':
              dispatch({
                type: 'STREAM_FINAL',
                text: event.text,
                stopped_reason: event.stopped_reason,
              });
              break;
            case 'interrupt': {
              const payload = event.args as unknown as InterruptPayload;
              dispatch({ type: 'STREAM_INTERRUPT', payload });
              break;
            }
            case 'error':
              dispatch({ type: 'STREAM_ERROR', text: event.text });
              break;
          }
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          dispatch({ type: 'STREAM_CANCELLED' });
        } else {
          dispatch({
            type: 'STREAM_ERROR',
            text: err instanceof Error ? err.message : 'Unknown error',
          });
        }
      }
    },
    [],
  );

  const sendMessage = useCallback(
    async (text: string) => {
      if (state.streamStatus === 'streaming') return;
      if (!text.trim()) return;

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const assistantId = crypto.randomUUID();
      dispatch({ type: 'SEND_USER_MESSAGE', text, assistantId });

      await _consumeStream(
        streamChat(
          { message: text, thread_id: state.threadId ?? undefined },
          controller.signal,
        ),
      );
    },
    [state.streamStatus, state.threadId, _consumeStream],
  );

  const resumeStream = useCallback(
    async (decision: Record<string, unknown>, displayText = '') => {
      if (!state.threadId) return;

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const assistantId = crypto.randomUUID();
      dispatch({ type: 'SEND_USER_MESSAGE', text: displayText, assistantId });

      await _consumeStream(
        resumeChat(state.threadId, decision, controller.signal),
      );
    },
    [state.threadId, _consumeStream],
  );

  const cancelStream = useCallback(() => {
    abortRef.current?.abort();
    if (state.threadId) {
      cancelChat(state.threadId).catch(() => { /* ignore */ });
    }
  }, [state.threadId]);

  const setInputValue = useCallback((value: string) => {
    dispatch({ type: 'SET_INPUT', value });
  }, []);

  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  return {
    messages: state.messages,
    streamStatus: state.streamStatus,
    currentTool: state.currentTool,
    threadId: state.threadId,
    inputValue: state.inputValue,
    errorMessage: state.errorMessage,
    interruptPayload: state.interruptPayload,
    sendMessage,
    resumeStream,
    cancelStream,
    setInputValue,
  };
}
