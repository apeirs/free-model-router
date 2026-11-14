// The three upstream wires, seen through one canonical form.
//
// A client request, whichever API it used, is first read as an OpenAI Chat Completions body (the
// canonical request). From it this module builds the request for an openai, anthropic or gemini provider,
// reads the provider's JSON answer back into one canonical result (text, tool calls, finish reason, usage),
// and renders that result as an OpenAI or Anthropic answer, as JSON or as server-sent events.
//
// Everything here is a pure function of its arguments: no clock, no randomness.

import { isRecord, type Json } from './providers.ts';
import { anthropicToOpenAI } from './wire.ts';

export interface ToolCall {
  id: string;
  name: string;
  /** The arguments as a JSON string. */
  args: string;
}

export interface Result {
  model: string;
  text: string;
  toolCalls: ToolCall[];
  /** The OpenAI finish_reason family: stop, length, tool_calls or content_filter. */
  finish: string;
  inputTokens: number;
  outputTokens: number;
}

const num = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The client's request as an OpenAI Chat Completions body, never streaming. */
export function canonical(api: 'openai' | 'anthropic', body: Json): Json {
  const out: Json = api === 'anthropic' ? anthropicToOpenAI(body, 'auto') : { ...body };
  delete out.stream;
  delete out.stream_options;
  return out;
}

/** Whether the canonical request asks for JSON mode. */
export function wantsJson(req: Json): boolean {
  const f = isRecord(req.response_format) ? req.response_format.type : undefined;
  return f === 'json_object' || f === 'json_schema';
}

/** Whether the canonical request carries an image. */
export function hasImage(req: Json): boolean {
  const messages = Array.isArray(req.messages) ? req.messages : [];
  return messages.some((m) => isRecord(m) && Array.isArray(m.content) && m.content.some((b) => isRecord(b) && b.type === 'image_url'));
}

export function hasTools(req: Json): boolean {
  return Array.isArray(req.tools) && req.tools.length > 0;
}

function maxTokensOf(req: Json): number | null {
  if (typeof req.max_tokens === 'number') return req.max_tokens;
  if (typeof req.max_completion_tokens === 'number') return req.max_completion_tokens;
  return null;
}

function stopsOf(req: Json): string[] {
  if (typeof req.stop === 'string') return [req.stop];
  if (Array.isArray(req.stop)) return req.stop.filter((s): s is string => typeof s === 'string');
  return [];
}

function parseArgs(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // not JSON: empty input
    }
  }
  return {};
}

/** A data: URL as its media type and base64 payload; null for any other URL. */
function dataUrl(url: string): { media: string; data: string } | null {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  return m === null ? null : { media: m[1] === '' ? 'image/png' : m[1], data: m[2] };
}

function imageUrlOf(block: Json): string | null {
  if (!isRecord(block.image_url)) return null;
  return typeof block.image_url.url === 'string' ? block.image_url.url : null;
}

// ---------------------------------------------------------------- requests

/** The openai-wire request: the canonical body naming the model. */
export function toOpenAI(req: Json, model: string): Json {
  return { ...req, model };
}

/** The anthropic-wire request. */
export function toAnthropic(req: Json, model: string): Json {
  const systems: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(req.messages) ? req.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system' || raw.role === 'developer') {
      const t = textOf(raw.content);
      if (t !== '') systems.push(t);
    } else if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: textOf(raw.content) }],
      });
    } else if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const t = textOf(raw.content);
      if (t !== '') blocks.push({ type: 'text', text: t });
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: String(call.function.name ?? ''), input: parseArgs(call.function.arguments) });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
    } else {
      if (typeof raw.content === 'string') {
        messages.push({ role: 'user', content: raw.content });
        continue;
      }
      const blocks: Json[] = [];
      for (const b of Array.isArray(raw.content) ? raw.content : []) {
        if (!isRecord(b)) continue;
        if (b.type === 'text' && typeof b.text === 'string') blocks.push({ type: 'text', text: b.text });
        else if (b.type === 'image_url') {
          const url = imageUrlOf(b);
          if (url === null) continue;
          const d = dataUrl(url);
          blocks.push({ type: 'image', source: d !== null ? { type: 'base64', media_type: d.media, data: d.data } : { type: 'url', url } });
        }
      }
      messages.push({ role: 'user', content: blocks });
    }
  }
  const out: Json = { model, messages };
  if (systems.length > 0) out.system = systems.join('\n');
  const max = maxTokensOf(req);
  if (max !== null) out.max_tokens = max;
  if (typeof req.temperature === 'number') out.temperature = req.temperature;
  if (typeof req.top_p === 'number') out.top_p = req.top_p;
  const stops = stopsOf(req);
  if (stops.length > 0) out.stop_sequences = stops;
  if (hasTools(req)) {
    out.tools = (req.tools as unknown[]).flatMap((t) => {
      if (!isRecord(t) || !isRecord(t.function) || typeof t.function.name !== 'string') return [];
      const f = t.function;
      return [{ name: f.name, description: typeof f.description === 'string' ? f.description : '', input_schema: isRecord(f.parameters) ? f.parameters : { type: 'object', properties: {} } }];
    });
    const c = req.tool_choice;
    if (c === 'auto') out.tool_choice = { type: 'auto' };
    else if (c === 'required') out.tool_choice = { type: 'any' };
    else if (isRecord(c) && isRecord(c.function) && typeof c.function.name === 'string') out.tool_choice = { type: 'tool', name: c.function.name };
  }
  return out;
}

/** A JSON schema with the keywords Gemini rejects removed. */
function geminiSchema(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(geminiSchema);
  if (!isRecord(value)) return value;
  const out: Json = {};
  for (const [k, v] of Object.entries(value)) {
    if (k === 'additionalProperties' || k === '$schema') continue;
    out[k] = geminiSchema(v);
  }
  return out;
}

/** The gemini-wire request body (the model is named in the path). */
export function toGemini(req: Json): Json {
  const contents: Json[] = [];
  const systems: string[] = [];
  const names = new Map<string, string>();
  const source = Array.isArray(req.messages) ? req.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system' || raw.role === 'developer') {
      const t = textOf(raw.content);
      if (t !== '') systems.push(t);
    } else if (raw.role === 'tool') {
      const id = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '';
      const t = textOf(raw.content);
      contents.push({ role: 'user', parts: [{ functionResponse: { name: names.get(id) ?? id, response: { result: t } } }] });
    } else if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const t = textOf(raw.content);
      if (t !== '') parts.push({ text: t });
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        const name = String(call.function.name ?? '');
        if (typeof call.id === 'string') names.set(call.id, name);
        parts.push({ functionCall: { name, args: parseArgs(call.function.arguments) } });
      }
      contents.push({ role: 'model', parts: parts.length > 0 ? parts : [{ text: '' }] });
    } else {
      const parts: Json[] = [];
      if (typeof raw.content === 'string') parts.push({ text: raw.content });
      else {
        for (const b of Array.isArray(raw.content) ? raw.content : []) {
          if (!isRecord(b)) continue;
          if (b.type === 'text' && typeof b.text === 'string') parts.push({ text: b.text });
          else if (b.type === 'image_url') {
            const url = imageUrlOf(b);
            if (url === null) continue;
            const d = dataUrl(url);
            parts.push(d !== null ? { inlineData: { mimeType: d.media, data: d.data } } : { fileData: { fileUri: url } });
          }
        }
      }
      contents.push({ role: 'user', parts });
    }
  }
  const out: Json = { contents };
  if (systems.length > 0) out.systemInstruction = { parts: [{ text: systems.join('\n') }] };
  const cfg: Json = {};
  const max = maxTokensOf(req);
  if (max !== null) cfg.maxOutputTokens = max;
  if (typeof req.temperature === 'number') cfg.temperature = req.temperature;
  if (typeof req.top_p === 'number') cfg.topP = req.top_p;
  const stops = stopsOf(req);
  if (stops.length > 0) cfg.stopSequences = stops;
  if (wantsJson(req)) cfg.responseMimeType = 'application/json';
  if (Object.keys(cfg).length > 0) out.generationConfig = cfg;
  if (hasTools(req)) {
    out.tools = [
      {
        functionDeclarations: (req.tools as unknown[]).flatMap((t) => {
          if (!isRecord(t) || !isRecord(t.function) || typeof t.function.name !== 'string') return [];
          const f = t.function;
          const decl: Json = { name: f.name, description: typeof f.description === 'string' ? f.description : '' };
          if (isRecord(f.parameters)) decl.parameters = geminiSchema(f.parameters);
          return [decl];
        }),
      },
    ];
    const c = req.tool_choice;
    if (c === 'required') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
    else if (isRecord(c) && isRecord(c.function) && typeof c.function.name === 'string') {
      out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [c.function.name] } };
    } else if (c === 'none') out.toolConfig = { functionCallingConfig: { mode: 'NONE' } };
  }
  return out;
}

// ---------------------------------------------------------------- answers

const finishOf = (reason: unknown): string => {
  switch (reason) {
    case 'length':
    case 'max_tokens':
    case 'MAX_TOKENS':
      return 'length';
    case 'tool_calls':
    case 'function_call':
    case 'tool_use':
      return 'tool_calls';
    case 'content_filter':
    case 'refusal':
    case 'SAFETY':
    case 'RECITATION':
    case 'BLOCKLIST':
    case 'PROHIBITED_CONTENT':
      return 'content_filter';
    default:
      return 'stop';
  }
};

export function fromOpenAI(body: Json, model: string): Result | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = choices[0];
  if (!isRecord(choice) || !isRecord(choice.message)) return null;
  const message = choice.message;
  const toolCalls: ToolCall[] = [];
  for (const [i, call] of (Array.isArray(message.tool_calls) ? message.tool_calls : []).entries()) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    const a = call.function.arguments;
    toolCalls.push({ id: typeof call.id === 'string' && call.id !== '' ? call.id : `call_${i}`, name: String(call.function.name ?? ''), args: typeof a === 'string' ? a : JSON.stringify(a ?? {}) });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    model: typeof body.model === 'string' && body.model !== '' ? body.model : model,
    text: typeof message.content === 'string' ? message.content : textOf(message.content),
    toolCalls,
    finish: finishOf(choice.finish_reason),
    inputTokens: num(usage.prompt_tokens),
    outputTokens: num(usage.completion_tokens),
  };
}

export function fromAnthropic(body: Json, model: string): Result | null {
  if (!Array.isArray(body.content)) return null;
  const toolCalls: ToolCall[] = [];
  for (const b of body.content) {
    if (isRecord(b) && b.type === 'tool_use') toolCalls.push({ id: typeof b.id === 'string' ? b.id : `toolu_${toolCalls.length}`, name: String(b.name ?? ''), args: JSON.stringify(isRecord(b.input) ? b.input : {}) });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    model: typeof body.model === 'string' && body.model !== '' ? body.model : model,
    text: textOf(body.content),
    toolCalls,
    finish: finishOf(body.stop_reason),
    inputTokens: num(usage.input_tokens),
    outputTokens: num(usage.output_tokens),
  };
}

export function fromGemini(body: Json, model: string): Result | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const cand = candidates[0];
  if (!isRecord(cand)) return null;
  const parts = isRecord(cand.content) && Array.isArray(cand.content.parts) ? cand.content.parts : [];
  let text = '';
  const toolCalls: ToolCall[] = [];
  for (const p of parts) {
    if (!isRecord(p)) continue;
    if (typeof p.text === 'string' && p.thought !== true) text += p.text;
    if (isRecord(p.functionCall)) toolCalls.push({ id: `call_${toolCalls.length}`, name: String(p.functionCall.name ?? ''), args: JSON.stringify(isRecord(p.functionCall.args) ? p.functionCall.args : {}) });
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  let finish = finishOf(cand.finishReason);
  if (toolCalls.length > 0 && finish === 'stop') finish = 'tool_calls';
  return {
    model: typeof body.modelVersion === 'string' && body.modelVersion !== '' ? body.modelVersion : model,
    text,
    toolCalls,
    finish,
    inputTokens: num(usage.promptTokenCount),
    outputTokens: num(usage.candidatesTokenCount),
  };
}

// ---------------------------------------------------------------- client renderings

const CREATED = 1767225600;
/** The characters in one streamed text delta. */
const DELTA_CHARS = 48;

function pieces(text: string): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += DELTA_CHARS) out.push(text.slice(at, at + DELTA_CHARS));
  return out;
}

const sse = (event: string | null, data: unknown): string => `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;

export function openAIJson(r: Result): Json {
  const message: Json = { role: 'assistant', content: r.text === '' && r.toolCalls.length > 0 ? null : r.text };
  if (r.toolCalls.length > 0) message.tool_calls = r.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.args } }));
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: CREATED,
    model: r.model,
    choices: [{ index: 0, message, finish_reason: r.finish, logprobs: null }],
    usage: { prompt_tokens: r.inputTokens, completion_tokens: r.outputTokens, total_tokens: r.inputTokens + r.outputTokens },
  };
}

export function openAISse(r: Result): string {
  const chunk = (delta: Json, finish: string | null): Json => ({ id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: CREATED, model: r.model, choices: [{ index: 0, delta, finish_reason: finish }] });
  const out = [sse(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const p of pieces(r.text)) out.push(sse(null, chunk({ content: p }, null)));
  for (const [i, c] of r.toolCalls.entries()) {
    out.push(sse(null, chunk({ tool_calls: [{ index: i, id: c.id, type: 'function', function: { name: c.name, arguments: c.args } }] }, null)));
  }
  out.push(sse(null, chunk({}, r.finish)));
  out.push(
    sse(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: CREATED,
      model: r.model,
      choices: [],
      usage: { prompt_tokens: r.inputTokens, completion_tokens: r.outputTokens, total_tokens: r.inputTokens + r.outputTokens },
    }),
  );
  out.push('data: [DONE]\n\n');
  return out.join('');
}

const anthropicStop = (finish: string): string => (finish === 'length' ? 'max_tokens' : finish === 'tool_calls' ? 'tool_use' : finish === 'content_filter' ? 'refusal' : 'end_turn');

export function anthropicJson(r: Result): Json {
  const content: Json[] = [];
  if (r.text !== '' || r.toolCalls.length === 0) content.push({ type: 'text', text: r.text });
  for (const c of r.toolCalls) content.push({ type: 'tool_use', id: c.id, name: c.name, input: parseArgs(c.args) });
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model: r.model,
    content,
    stop_reason: anthropicStop(r.finish),
    stop_sequence: null,
    usage: { input_tokens: r.inputTokens, output_tokens: r.outputTokens },
  };
}

export function anthropicSse(r: Result): string {
  const out = [
    sse('message_start', {
      type: 'message_start',
      message: { id: 'msg_agg', type: 'message', role: 'assistant', model: r.model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: r.inputTokens, output_tokens: 0 } },
    }),
  ];
  let index = 0;
  if (r.text !== '' || r.toolCalls.length === 0) {
    out.push(sse('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const p of pieces(r.text)) out.push(sse('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: p } }));
    out.push(sse('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const c of r.toolCalls) {
    out.push(sse('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id: c.id, name: c.name, input: {} } }));
    out.push(sse('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: c.args } }));
    out.push(sse('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  out.push(sse('message_delta', { type: 'message_delta', delta: { stop_reason: anthropicStop(r.finish), stop_sequence: null }, usage: { output_tokens: r.outputTokens } }));
  out.push(sse('message_stop', { type: 'message_stop' }));
  return out.join('');
}
