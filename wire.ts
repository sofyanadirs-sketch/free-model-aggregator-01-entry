// The client wires and the three upstream wires.
//
// The trial's client calls this server as an OpenAI service (POST
// /v1/chat/completions) or an Anthropic one (POST /v1/messages), with a JSON body
// or with server-sent events. Each provider speaks its own wire — OpenAI,
// Anthropic or Gemini — and this file translates between them.
//
// The translation is built around one normalized answer: every provider's reply,
// JSON or streamed, is reduced to its text and its usage counts, and the answer
// is then rendered in the wire the client used. That keeps the two things the
// client checks — the provider's text whole, and its usage counts exactly as
// the provider reported them — the same whatever wire served the request.
//
// What must survive a translation untouched, because the provider echoes what it
// parsed and the client compares that echo with what it sent: the system prompt,
// the user text, the images, the tool names, the stop sequences, max_tokens and
// JSON mode. A JSON-mode request never goes to an anthropic-wire provider: that
// wire cannot carry it, so it would echo that JSON mode was absent.
//
// Everything here is a pure function of its arguments: no clock and no
// randomness, so the same body and the same provider answer give the same bytes
// on a replay. The two model listings use fixed timestamps for the same reason.

import { authHeaders, isRecord, joinBase, type Json, type ProviderDirectoryEntry } from './providers.ts';

/** One model this entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The id a rendered OpenAI answer carries. */
const OPENAI_ANSWER_ID = 'chatcmpl-aggregate';
/** The id a rendered Anthropic answer carries. */
const ANTHROPIC_ANSWER_ID = 'msg_aggregate';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
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

/** A number, or 0 when the value is not one. */
function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
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

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
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

/** An OpenAI image_url part as a Gemini part: inline data, or a file reference for a plain URL. */
function geminiImagePart(imageUrl: unknown): Json | null {
  if (!isRecord(imageUrl) || typeof imageUrl.url !== 'string') return null;
  const url = imageUrl.url;
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
  if (/^https?:\/\//i.test(url)) return { fileData: { fileUri: url } };
  return null;
}

/** An Anthropic image block as a Gemini part: inline data, or a file reference for a plain URL. */
function geminiImageFromAnthropic(block: Json): Json | null {
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  return null;
}

/** An OpenAI content part as a Gemini part: text or an image; null for the rest. */
function geminiPartOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url') return geminiImagePart(part.image_url);
  return null;
}

/** The system text of an OpenAI body: every system message's text, joined with a newline. */
function openAISystemText(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const raw of messages) {
    if (!isRecord(raw) || raw.role !== 'system') continue;
    const text = textOf(raw.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

/** The system text of an Anthropic body: its system prompt, string or blocks. */
function anthropicSystemText(body: Json): string {
  return textOf(body.system);
}

/** OpenAI tools as Gemini function declarations. */
function geminiDeclarationsFromOpenAI(body: Json): Json[] {
  const tools = Array.isArray(body.tools) ? body.tools : [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
}

/** Anthropic tools as Gemini function declarations. */
function geminiDeclarationsFromAnthropic(body: Json): Json[] {
  const tools = Array.isArray(body.tools) ? body.tools : [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
  });
}

/** The generation config of a Gemini request: the output cap, stops, sampling and JSON mode. */
function geminiConfig(maxTokens: number | null, stops: string[], temperature: unknown, topP: unknown, jsonMode: boolean): Json {
  const config: Json = {};
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof temperature === 'number') config.temperature = temperature;
  if (typeof topP === 'number') config.topP = topP;
  if (jsonMode) config.responseMimeType = 'application/json';
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request: the
 * system prompt becomes the system instruction, user and assistant messages
 * become user and model contents (images as inline data, tool calls as function
 * calls), tool messages fold into the previous user content as function
 * responses so they never overwrite the user text, and tools become function
 * declarations.
 */
export function openAIToGemini(body: Json, jsonMode: boolean): Json {
  const contents: Json[] = [];
  const system = openAISystemText(body);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'tool') {
      // A tool result answers a call: keep it out of the user text by folding it
      // into the previous user content as a function response.
      const response = { functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : 'tool', response: { content: toolResultText(raw.content) } } };
      const last = contents.length > 0 ? contents[contents.length - 1] : null;
      if (last !== null && last.role === 'user' && Array.isArray(last.parts)) last.parts.push(response);
      else contents.push({ role: 'user', parts: [response] });
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        const part = geminiPartOf(block);
        if (part !== null) parts.push(part);
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  if (contents.length === 0) contents.push({ role: 'user', parts: [{ text: '' }] });

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = geminiDeclarationsFromOpenAI(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  const stops = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : [];
  const config = geminiConfig(maxTokens, stops, body.temperature, body.top_p, jsonMode);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request: the system
 * prompt becomes the system instruction, user messages become user contents
 * (images as inline data, tool results as function responses), assistant
 * messages become model contents (tool uses as function calls).
 */
export function anthropicToGemini(body: Json, jsonMode: boolean): Json {
  const contents: Json[] = [];
  const system = anthropicSystemText(body);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image') {
          const part = geminiImageFromAnthropic(block);
          if (part !== null) parts.push(part);
        } else if (block.type === 'tool_result') {
          parts.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : 'tool', response: { content: toolResultText(block.content) } } });
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  if (contents.length === 0) contents.push({ role: 'user', parts: [{ text: '' }] });

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = geminiDeclarationsFromAnthropic(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : null;
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [];
  const config = geminiConfig(maxTokens, stops, body.temperature, body.top_p, jsonMode);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names
 * `model`: the system prompt becomes a system message, image blocks become
 * image_url data URLs, tool_use blocks become tool_calls, tool_result blocks
 * become tool messages, tools and tool_choice are rewritten, and the output cap,
 * temperature, top_p, stop sequences and streaming pass through. JSON mode is
 * carried as response_format when the caller asks for it.
 */
export function anthropicToOpenAI(body: Json, model: string, jsonMode: boolean): Json {
  const messages: Json[] = [];
  const system = anthropicSystemText(body);
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
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (jsonMode) out.response_format = { type: 'json_object' };
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names
 * `model`: system messages become the system prompt, image_url parts become
 * image blocks, tool_calls become tool_use blocks, tool messages become
 * tool_result blocks folded into the user message that follows them (or the
 * last user message, so they never overwrite the user text), and tools,
 * tool_choice, the output cap, temperature, top_p, stop sequences and streaming
 * pass through. JSON mode cannot be carried on this wire, so the caller never
 * routes a JSON-mode request here.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const system = openAISystemText(body);
  const messages: Json[] = [];
  const pendingResults: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'tool') {
      pendingResults.push({
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: toolResultText(raw.content),
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    const blocks: Json[] = [...pendingResults];
    pendingResults.length = 0;
    if (typeof raw.content === 'string') {
      if (raw.content !== '') blocks.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
          if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
          else blocks.push({ type: 'image', source: { type: 'url', url } });
        }
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks.length === 1 && blocks[0]?.type === 'text' ? (blocks[0] as Json).text : blocks });
  }
  if (pendingResults.length > 0) {
    const last = messages.length > 0 ? messages[messages.length - 1] : null;
    if (last !== null && last.role === 'user') {
      const content = Array.isArray(last.content) ? last.content : typeof last.content === 'string' ? [{ type: 'text', text: last.content }] : [];
      last.content = [...content, ...pendingResults];
    } else {
      messages.push({ role: 'user', content: pendingResults });
    }
  }

  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'required') out.tool_choice = { type: 'any' };
    else if (choice === 'none') out.tool_choice = { type: 'none' };
  } else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One provider call: the URL, the headers and the JSON body to send. */
export interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: string;
}

/**
 * The client's request rewritten for one provider's wire. `clientApi` is the wire
 * the client used, `jsonMode` says the request asks for JSON mode (from the
 * manifest's requirements or the client's own response_format).
 */
export function upstreamCall(
  envUrl: string,
  provider: ProviderDirectoryEntry,
  model: string,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  jsonMode: boolean,
): UpstreamCall {
  const base = joinBase(envUrl, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'openai') {
    let request: Json;
    if (clientApi === 'openai') {
      request = { ...body, model };
      if (body.stream === true) request.stream_options = { include_usage: true };
    } else {
      request = anthropicToOpenAI(body, model, jsonMode);
    }
    if (body.stream === true) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/chat/completions`, headers, body: JSON.stringify(request) };
  }
  if (provider.wire === 'anthropic') {
    const request = clientApi === 'anthropic' ? { ...body, model } : openAIToAnthropic(body, model);
    headers['anthropic-version'] = '2023-06-01';
    if (body.stream === true) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/messages`, headers, body: JSON.stringify(request) };
  }
  const request = clientApi === 'anthropic' ? anthropicToGemini(body, jsonMode) : openAIToGemini(body, jsonMode);
  const stream = body.stream === true;
  const path = stream ? `/v1beta/models/${encodeURIComponent(model)}:streamGenerateContent?alt=sse` : `/v1beta/models/${encodeURIComponent(model)}:generateContent`;
  if (stream) headers.accept = 'text/event-stream';
  return { url: `${base}${path}`, headers, body: JSON.stringify(request) };
}

/** One provider's answer, reduced to its text and its usage counts. */
export interface NormalizedAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

/** The text of a Chat Completions message: a string, or its text blocks joined. */
function openAIText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** The text of an Anthropic message: its text blocks joined, nothing else. */
function anthropicText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** The text of a Gemini candidate: its parts' text, joined. */
function geminiText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** One SSE frame: its event name, if any, and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split a server-sent event stream into frames. */
export function parseSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const raw of normalized.split('\n\n')) {
    if (raw.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** A frame's data as a JSON object, or null when it is not one. */
function frameJson(frame: SseFrame): Json | null {
  if (frame.data.trim() === '[DONE]') return null;
  try {
    const value: unknown = JSON.parse(frame.data);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** An OpenAI JSON answer as a normalized answer. */
function openAIJsonAnswer(body: Json): NormalizedAnswer | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text: openAIText(message.content), inputTokens: numberOrZero(usage.prompt_tokens), outputTokens: numberOrZero(usage.completion_tokens) };
}

/** An OpenAI SSE stream as a normalized answer: content deltas joined, usage from the usage chunk. */
function openAIStreamAnswer(text: string): NormalizedAnswer | null {
  let out = '';
  let input = 0;
  let output = 0;
  let sawAny = false;
  for (const frame of parseSse(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    const chunk = frameJson(frame);
    if (chunk === null) continue;
    sawAny = true;
    if (isRecord(chunk.usage)) {
      input = numberOrZero(chunk.usage.prompt_tokens);
      output = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
  }
  return sawAny ? { text: out, inputTokens: input, outputTokens: output } : null;
}

/** An Anthropic JSON answer as a normalized answer. */
function anthropicJsonAnswer(body: Json): NormalizedAnswer | null {
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text: anthropicText(body.content), inputTokens: numberOrZero(usage.input_tokens), outputTokens: numberOrZero(usage.output_tokens) };
}

/** An Anthropic SSE stream as a normalized answer: text deltas joined, usage from message_start and message_delta. */
function anthropicStreamAnswer(text: string): NormalizedAnswer | null {
  let out = '';
  let input = 0;
  let output = 0;
  let sawAny = false;
  for (const frame of parseSse(text)) {
    const data = frameJson(frame);
    if (data === null) continue;
    sawAny = true;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    }
    if (frame.event === 'message_delta' && isRecord(data.usage)) {
      output = numberOrZero(data.usage.output_tokens);
    }
  }
  return sawAny ? { text: out, inputTokens: input, outputTokens: output } : null;
}

/** A Gemini JSON answer as a normalized answer. */
function geminiJsonAnswer(body: Json): NormalizedAnswer | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { text: geminiText(content.parts), inputTokens: numberOrZero(usage.promptTokenCount), outputTokens: numberOrZero(usage.candidatesTokenCount) };
}

/** A Gemini SSE stream as a normalized answer: candidate parts joined, usage from the final frame. */
function geminiStreamAnswer(text: string): NormalizedAnswer | null {
  let out = '';
  let input = 0;
  let output = 0;
  let sawAny = false;
  for (const frame of parseSse(text)) {
    const data = frameJson(frame);
    if (data === null) continue;
    sawAny = true;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null && isRecord(candidate.content)) out += geminiText(candidate.content.parts);
    if (isRecord(data.usageMetadata)) {
      input = numberOrZero(data.usageMetadata.promptTokenCount);
      output = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  return sawAny ? { text: out, inputTokens: input, outputTokens: output } : null;
}

/**
 * One provider's reply as a normalized answer, or null when it carries none.
 * `stream` is whether the call asked for server-sent events.
 */
export function parseUpstreamAnswer(wire: 'openai' | 'anthropic' | 'gemini', stream: boolean, text: string): NormalizedAnswer | null {
  if (stream) {
    if (wire === 'openai') return openAIStreamAnswer(text);
    if (wire === 'anthropic') return anthropicStreamAnswer(text);
    return geminiStreamAnswer(text);
  }
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(body)) return null;
  if (wire === 'openai') return openAIJsonAnswer(body);
  if (wire === 'anthropic') return anthropicJsonAnswer(body);
  return geminiJsonAnswer(body);
}

/** The message a provider's error body carries, or '' when it carries none. */
export function errorMessageOf(wire: 'openai' | 'anthropic' | 'gemini', text: string): string {
  try {
    const body: unknown = JSON.parse(text);
    if (!isRecord(body)) return '';
    const error = isRecord(body.error) ? body.error : null;
    if (error !== null && typeof error.message === 'string') return error.message;
    if (typeof body.message === 'string') return body.message;
    return '';
  } catch {
    return '';
  }
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into pieces of at most `size` characters, for streamed deltas. */
function splitPieces(text: string, size: number): string[] {
  if (text === '') return [];
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += size) pieces.push(text.slice(at, at + size));
  return pieces;
}

/** A normalized answer as a Chat Completions object. */
export function renderOpenAIJson(answer: NormalizedAnswer, model: string): Json {
  return {
    id: OPENAI_ANSWER_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** A normalized answer as a Chat Completions event stream: role, content, finish, usage, then [DONE]. */
export function renderOpenAIStream(answer: NormalizedAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: OPENAI_ANSWER_ID,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitPieces(answer.text, 64)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, 'stop')));
  frames.push(
    sseEvent(null, {
      id: OPENAI_ANSWER_ID,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalized answer as an Anthropic Messages object. */
export function renderAnthropicJson(answer: NormalizedAnswer, model: string): Json {
  return {
    id: ANTHROPIC_ANSWER_ID,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A normalized answer as an Anthropic event stream: message_start, one text block, message_delta, message_stop. */
export function renderAnthropicStream(answer: NormalizedAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: ANTHROPIC_ANSWER_ID,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitPieces(answer.text, 64)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: answer.outputTokens } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
