import { useCallback, useReducer, useRef } from 'react';
import { streamChat, cancelChat } from '@/api';
import {
  isConversationIdPayload,
  type AssistantMessage,
  type ConversationMessage,
  type MessageBlock,
  type StreamStatus,
  type ThinkingBlock,
} from '@/types';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

interface ChatState {
  conversationId: string | null;
  messages: ConversationMessage[];
  streamStatus: StreamStatus;
  currentTool: string | null;
  inputValue: string;
}

const initialState: ChatState = {
  conversationId: null,
  messages: [],
  streamStatus: 'idle',
  currentTool: null,
  inputValue: '',
};

// ---------------------------------------------------------------------------
// Action union
// ---------------------------------------------------------------------------

type ChatAction =
  | { type: 'SET_INPUT'; value: string }
  | { type: 'SEND_USER_MESSAGE'; text: string; assistantId: string }
  | { type: 'SET_CONVERSATION_ID'; id: string }
  | { type: 'APPEND_THINKING'; text: string }
  | { type: 'ADD_TOOL_CALL'; name: string; args: Record<string, unknown>; call_id: string }
  | { type: 'ADD_TOOL_RESULT'; text: string; name: string; call_id: string; is_error: boolean }
  | { type: 'STREAM_DONE'; text: string; stopped_reason: string }
  | { type: 'STREAM_ERROR'; text: string }
  | { type: 'STREAM_CANCELLED' };

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getLastAssistantMessage(messages: ConversationMessage[]): AssistantMessage | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === 'assistant' && msg.isStreaming) {
      return msg;
    }
  }
  return null;
}

/**
 * Returns a new messages array where the last streaming AssistantMessage has
 * been replaced with the result of `updater`. If no such message exists the
 * original array is returned unchanged.
 */
function updateLastStreamingMessage(
  messages: ConversationMessage[],
  updater: (msg: AssistantMessage) => AssistantMessage,
): ConversationMessage[] {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === 'assistant' && msg.isStreaming) {
      const updated = updater(msg);
      return [...messages.slice(0, i), updated, ...messages.slice(i + 1)];
    }
  }
  return messages;
}

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case 'SET_INPUT':
      return { ...state, inputValue: action.value };

    case 'SEND_USER_MESSAGE': {
      const userMsg: ConversationMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        text: action.text,
      };
      const assistantMsg: AssistantMessage = {
        id: action.assistantId,
        role: 'assistant',
        blocks: [],
        isStreaming: true,
      };
      return {
        ...state,
        messages: [...state.messages, userMsg, assistantMsg],
        streamStatus: 'streaming',
        inputValue: '',
        currentTool: null,
      };
    }

    case 'SET_CONVERSATION_ID':
      return { ...state, conversationId: action.id };

    case 'APPEND_THINKING': {
      const messages = updateLastStreamingMessage(state.messages, (msg) => {
        const lastBlock = msg.blocks[msg.blocks.length - 1];
        let newBlocks: MessageBlock[];
        if (lastBlock && lastBlock.kind === 'thinking') {
          // Append to existing ThinkingBlock — produce a new block object
          const updatedBlock: ThinkingBlock = { ...lastBlock, text: lastBlock.text + action.text };
          newBlocks = [...msg.blocks.slice(0, -1), updatedBlock];
        } else {
          // Push a new ThinkingBlock
          const newBlock: ThinkingBlock = { kind: 'thinking', text: action.text };
          newBlocks = [...msg.blocks, newBlock];
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

    case 'STREAM_DONE': {
      const lastMsg = getLastAssistantMessage(state.messages);
      const lastThinkingBlock = lastMsg?.blocks
        .filter((b): b is ThinkingBlock => b.kind === 'thinking')
        .at(-1);

      const shouldSetFinalText =
        action.text !== '' && action.text !== lastThinkingBlock?.text;

      const messages = updateLastStreamingMessage(state.messages, (msg) => ({
        ...msg,
        isStreaming: false,
        ...(shouldSetFinalText ? { finalText: action.text } : {}),
      }));
      return { ...state, messages, streamStatus: 'done', currentTool: null };
    }

    case 'STREAM_ERROR': {
      const messages = updateLastStreamingMessage(state.messages, (msg) => ({
        ...msg,
        isStreaming: false,
      }));
      return { ...state, messages, streamStatus: 'error', currentTool: null };
    }

    case 'STREAM_CANCELLED': {
      const messages = updateLastStreamingMessage(state.messages, (msg) => ({
        ...msg,
        isStreaming: false,
      }));
      return { ...state, messages, streamStatus: 'idle', currentTool: null };
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

  const sendMessage = useCallback(
    async (text: string) => {
      if (state.streamStatus === 'streaming') return;
      if (!text.trim()) return;

      // Cancel any lingering previous request
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const assistantId = crypto.randomUUID();
      dispatch({ type: 'SEND_USER_MESSAGE', text, assistantId });

      try {
        for await (const event of streamChat(
          { message: text, conversation_id: state.conversationId ?? undefined },
          controller.signal,
        )) {
          if (isConversationIdPayload(event)) {
            dispatch({ type: 'SET_CONVERSATION_ID', id: event.conversation_id });
            continue;
          }

          // Narrowed to AgentEvent by isAgentEvent in api.ts's streamChat generator
          switch (event.type) {
            case 'thinking':
              dispatch({ type: 'APPEND_THINKING', text: event.text });
              break;
            case 'tool_call':
              dispatch({ type: 'ADD_TOOL_CALL', name: event.name, args: event.args, call_id: event.call_id });
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
            case 'done':
              dispatch({ type: 'STREAM_DONE', text: event.text, stopped_reason: event.stopped_reason });
              break;
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
    [state.streamStatus, state.conversationId],
  );

  const cancelStream = useCallback(() => {
    abortRef.current?.abort();
    // Fire-and-forget backend cancel
    if (state.conversationId) {
      cancelChat(state.conversationId).catch(() => {
        /* ignore */
      });
    }
  }, [state.conversationId]);

  const setInputValue = useCallback((value: string) => {
    dispatch({ type: 'SET_INPUT', value });
  }, []);

  return {
    messages: state.messages,
    streamStatus: state.streamStatus,
    currentTool: state.currentTool,
    conversationId: state.conversationId,
    inputValue: state.inputValue,
    sendMessage,
    cancelStream,
    setInputValue,
  };
}
