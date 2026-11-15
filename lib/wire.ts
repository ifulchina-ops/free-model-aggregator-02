// The client wires, the one upstream wire, and the model field.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages), with a JSON body or with server-sent events. The entry dials the grader's
// gateway, an OpenAI-compatible endpoint, so every prompt goes upstream as a Chat Completions request
// and the Anthropic path is translated both ways.
//
// Unlike a relay, this entry always dials the gateway NON-streaming, even when the client asked for
// server-sent events: the complete answer is verified first (it must carry the prompt's token), and
// the stream is then synthesized from it in the exact event order each wire expects. That keeps a
// malformed provider stream from ever reaching the client, and a prompt whose first model answered
// empty can still fail over before anything is sent.
//
// THE MODEL FIELD
//   The house records every gateway call and scores the model that really answered, so a response's
//   `model` must be the model this entry dialled. OpenRouter's own answer names a free model without
//   the gateway's `openrouter/` prefix; these functions put the dialled id back on every answer and
//   every streamed chunk. They never make a model up: the id came from the brief and went to the
//   gateway in the request's `model`.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body
// and the same gateway answer give the same bytes on a replay. The two model listings use fixed
// timestamps for the same reason: a `created` that moved between runs would be a nondeterministic
// document. Stream chunks reuse the provider answer's own id and created timestamp.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The answer text of a Chat Completions body: the first choice's message content as text. */
export function completionText(body: Json): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  return textOf(message.content);
}

/** The first choice of a Chat Completions body, or {}. */
function firstChoice(body: Json): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  return isRecord(choices[0]) ? choices[0] : {};
}

/** The finish reason of a Chat Completions body, or 'stop' when it names none. */
function finishReasonOf(body: Json): string {
  const finish = firstChoice(body).finish_reason;
  return typeof finish === 'string' && finish !== '' ? finish : 'stop';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The usage of a Chat Completions body as {prompt_tokens, completion_tokens}. */
function usageOf(body: Json): { prompt_tokens: number; completion_tokens: number } {
  const usage = isRecord(body.usage) ? body.usage : {};
  return { prompt_tokens: numberOrZero(usage.prompt_tokens), completion_tokens: numberOrZero(usage.completion_tokens) };
}

/** The tool calls of a Chat Completions body, as OpenAI shapes them. */
function toolCallsOf(body: Json): Json[] {
  const message = isRecord(firstChoice(body).message) ? firstChoice(body).message : {};
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  return calls.filter((c): c is Json => isRecord(c));
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences pass through. Streaming is never requested upstream:
 * the entry dials non-streaming and synthesizes the stream from the complete answer.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const message = isRecord(firstChoice(body).message) ? firstChoice(body).message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  for (const call of toolCallsOf(body)) {
    if (!isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = usageOf(body);
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(firstChoice(body).finish_reason),
    stop_sequence: null,
    usage: { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
  };
}

/** A Chat Completions answer with `model` set to the id the entry dialled: the one truthful claim. */
export function openAIJsonWithModel(body: Json, model: string): Json {
  return { ...body, model };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** The response id of a Chat Completions body, or a fixed fallback. */
function answerId(body: Json): string {
  return typeof body.id === 'string' && body.id !== '' ? body.id : 'chatcmpl-aggregate';
}

/** The created timestamp of a Chat Completions body, or a fixed fallback. */
function answerCreated(body: Json): number {
  return typeof body.created === 'number' && Number.isFinite(body.created) ? body.created : FIXED_OPENAI_CREATED;
}

/**
 * A complete Chat Completions answer as an OpenAI SSE stream: a role chunk, the text in one content
 * chunk, one finish chunk carrying the usage, then [DONE]. The text is split at most once: the env
 * reads the concatenated deltas, and one content chunk keeps the order trivially right.
 * `model` is the id the entry dialled, named on every chunk.
 */
export function openAIStreamFromAnswer(body: Json, model: string): string {
  const id = answerId(body);
  const created = answerCreated(body);
  const text = completionText(body);
  const finish = finishReasonOf(body);
  const usage = usageOf(body);
  const calls = toolCallsOf(body);
  const frames: string[] = [];
  const roleDelta: Json = { role: 'assistant', content: '' };
  frames.push(
    sseEvent(null, {
      id,
      object: 'chat.completion.chunk',
      created,
      model,
      choices: [{ index: 0, delta: roleDelta, finish_reason: null }],
    }),
  );
  if (text !== '') {
    frames.push(
      sseEvent(null, {
        id,
        object: 'chat.completion.chunk',
        created,
        model,
        choices: [{ index: 0, delta: { content: text }, finish_reason: null }],
      }),
    );
  }
  for (let i = 0; i < calls.length; i++) {
    const call = calls[i];
    const fn = isRecord(call.function) ? call.function : {};
    frames.push(
      sseEvent(null, {
        id,
        object: 'chat.completion.chunk',
        created,
        model,
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: i,
                  id: typeof call.id === 'string' ? call.id : '',
                  type: 'function',
                  function: {
                    name: typeof fn.name === 'string' ? fn.name : '',
                    arguments: typeof fn.arguments === 'string' ? fn.arguments : JSON.stringify(fn.arguments ?? {}),
                  },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      }),
    );
  }
  frames.push(
    sseEvent(null, {
      id,
      object: 'chat.completion.chunk',
      created,
      model,
      choices: [{ index: 0, delta: {}, finish_reason: finish }],
      usage: { prompt_tokens: usage.prompt_tokens, completion_tokens: usage.completion_tokens, total_tokens: usage.prompt_tokens + usage.completion_tokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/**
 * A complete Chat Completions answer as the Anthropic event sequence: message_start, one text block
 * (start/delta/stop) when there is text, one tool_use block per tool call, then message_delta with
 * the stop reason and output tokens, then message_stop. `model` is the id the entry dialled.
 */
export function anthropicStreamFromAnswer(body: Json, model: string): string {
  const usage = usageOf(body);
  const text = completionText(body);
  const calls = toolCallsOf(body);
  const stopReason = stopReasonOf(firstChoice(body).finish_reason);
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: `msg_${answerId(body)}`,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.prompt_tokens, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  if (text !== '') {
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text } }));
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of calls) {
    const fn = isRecord(call.function) ? call.function : {};
    const name = typeof fn.name === 'string' ? fn.name : '';
    const args = typeof fn.arguments === 'string' ? fn.arguments : JSON.stringify(fn.arguments ?? {});
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`, name, input: {} },
      }),
    );
    if (args !== '') {
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: args } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: usage.completion_tokens } }),
  );
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}
