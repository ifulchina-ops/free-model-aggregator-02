// The entry: one server over the free models the brief lists.
//
//   POST /v1/chat/completions  the OpenAI wire, JSON and server-sent events
//   POST /v1/messages          the Anthropic wire, JSON and server-sent events
//   GET  /v1/models            the served models, in the wire its auth header names
//   node main.ts --list        prints the catalog and exits
//
// Every prompt is served by the best free model that meets its requirements (the brief's rating,
// milli-points, highest first), dialled on the grader's gateway as a Chat Completions call. When a
// model refuses or answers without the prompt's token, the entry fails over to the next-best model
// that meets the requirements, and the refusal is reported on the response as the
// `x-makefun-upstream-failures` trail header: the failover column is only visible through it.
//
// The entry always dials the gateway NON-streaming, even when the client asked for a stream: the
// complete answer is verified first (non-empty, and carrying the prompt's token when the request
// names one), and the stream is then synthesized from it. A prompt whose first model answered empty
// can still fail over before anything reaches the client, and a malformed provider stream never does.
//
// Every reply's `model` is the model the entry dialled: the house records every gateway call and
// scores the model that really answered. The entry reports only the model it sent to the gateway.
//
// The entry is deterministic: no clock and no randomness in anything it sends. Request timeouts use
// the clock only to budget a prompt's 28 s, never in a reply.
//
// The entry knows one provider: OpenRouter. The gateway allowlists `openrouter/*:free` only, and the
// brief's free model list is exactly those fourteen ids; the catalog lists them and nothing else.
//
//   node main.ts            start the server (in a trial: register, then poll for work)
//   node main.ts --list     print the catalog this entry serves and exit
//
// Environment: MAKEFUN_PROMPT_FILE (the brief), MAKEFUN_ENV_URL (register/poll), OPENAI_BASE_URL,
// OPENAI_API_KEY (the gateway's Chat Completions route and its dummy key; the gateway swaps the key
// for the OpenRouter key pushed as a secret).

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import {
  readBrief,
  freeModels,
  rankedCandidates,
  buildCatalog,
  capabilitiesOf,
  meetsRequirements,
  type Brief,
  type ModelRow,
  type PromptRow,
  type Requirement,
  type Json,
  isRecord,
} from './lib/providers.ts';
import {
  openAIModelList,
  anthropicModelList,
  anthropicError,
  openAIError,
  completionText,
  anthropicToOpenAI,
  openAIToAnthropic,
  openAIJsonWithModel,
  openAIStreamFromAnswer,
  anthropicStreamFromAnswer,
} from './lib/wire.ts';

/** How long one prompt may take, wall clock, from its arrival. The env caps a prompt at 30 s. */
const PROMPT_TIMEOUT_MS = 28_000;
/** How long one upstream attempt may take; the prompt deadline cuts it shorter near the end. */
const ATTEMPT_TIMEOUT_MS = 15_000;
/** The largest request body the entry accepts. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The gateway requires max_tokens on every call; when the client sends none, this is it. */
const DEFAULT_MAX_TOKENS = 512;
/** No upstream call asks for more than this many tokens. */
const MAX_TOKENS_CAP = 8192;
/** How many upstream calls one prompt may spend before giving up. The trial cap is 60; six prompts. */
const MAX_ATTEMPTS_PER_PROMPT = 8;
/** How often the entry asks the env for work, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The trail header carries at most this many refused attempts. */
const MAX_TRAIL_ROWS = 8;

/** One refused upstream attempt, as the trail header reports it. */
interface UpstreamFailure {
  status: number;
  code: string;
  model: string;
}

/** The outcome of serving one prompt. */
interface ServedPrompt {
  answer: Json;
  model: string;
  trail: UpstreamFailure[];
}

/** The outcome when no model served a prompt: the refusals, for the trail header. */
interface UnservedPrompt {
  answer: null;
  trail: UpstreamFailure[];
}

const briefPath = process.env.MAKEFUN_PROMPT_FILE ?? '';
const envUrl = process.env.MAKEFUN_ENV_URL ?? '';
const openAIBase = process.env.OPENAI_BASE_URL ?? '';
const openAIKey = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(briefPath);
if (brief.error !== null) {
  console.error(`[aggregator] brief: ${brief.error}`);
  // In a trial the brief is the work; without it there is nothing to serve. Outside one, the server
  // still starts (an empty catalog, nothing to serve) so --list and local probes behave.
  if (envUrl !== '') process.exit(1);
}

const catalog = buildCatalog(brief);
const servedModels = freeModels(brief).map((m) => ({ id: m.id, provider: m.provider, display_name: m.display_name }));

/** Models that refused or answered badly in this trial: skipped for the rest of it. */
const cooledDown = new Set<string>();

/** The prompt index, in the manifest's order: the env sends the six prompts one at a time. */
let promptIndex = 0;

/** The word the prompt's token takes: CAPITAL letters, a hyphen, four digits (ZEBRA-4721). */
const TOKEN_PATTERN = /[A-Z]+-[0-9]{4}/;

/** The token a request names, in its system or user text, or null when it names none. */
function tokenOf(body: Json, api: 'openai' | 'anthropic'): string | null {
  const texts: string[] = [];
  if (api === 'anthropic') {
    if (typeof body.system === 'string') texts.push(body.system);
    if (Array.isArray(body.system)) {
      for (const block of body.system) if (isRecord(block) && typeof block.text === 'string') texts.push(block.text);
    }
    if (Array.isArray(body.messages)) {
      for (const message of body.messages) {
        if (!isRecord(message)) continue;
        if (typeof message.content === 'string') texts.push(message.content);
        if (Array.isArray(message.content)) {
          for (const block of message.content) if (isRecord(block) && typeof block.text === 'string') texts.push(block.text);
        }
      }
    }
  } else if (Array.isArray(body.messages)) {
    for (const message of body.messages) {
      if (!isRecord(message)) continue;
      if (typeof message.content === 'string') texts.push(message.content);
      if (Array.isArray(message.content)) {
        for (const part of message.content) if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
      }
    }
  }
  for (const text of texts) {
    const match = TOKEN_PATTERN.exec(text);
    if (match !== null) return match[0];
  }
  return null;
}

/** What a request needs, read off its own body: JSON mode, tools, or a long context. */
function requirementsFromBody(body: Json, api: 'openai' | 'anthropic'): Requirement {
  const requires: Requirement = {};
  if (api === 'openai') {
    if (isRecord(body.response_format) && body.response_format.type === 'json_object') requires.json = true;
    if (Array.isArray(body.tools) && body.tools.length > 0) requires.tools = true;
  } else {
    if (Array.isArray(body.tools) && body.tools.length > 0) requires.tools = true;
  }
  return requires;
}

/** The max_tokens a gateway call carries: the client's, clamped, or the default when it sends none. */
function maxTokensOf(body: Json): number {
  const value = typeof body.max_tokens === 'number' && Number.isFinite(body.max_tokens) ? Math.floor(body.max_tokens) : DEFAULT_MAX_TOKENS;
  return Math.max(1, Math.min(MAX_TOKENS_CAP, value));
}

/** The full request body as JSON, or null when it is not one (or over the size cap). */
async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = chunk as Buffer;
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  try {
    const parsed: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

/** The gateway's Chat Completions route. */
function gatewayCompletionsUrl(): string {
  return `${openAIBase.replace(/\/$/, '')}/chat/completions`;
}

/** One upstream call: POST the Chat Completions body to the gateway, returning its JSON on success. */
async function dial(model: string, upstream: Json, timeoutMs: number): Promise<{ ok: true; body: Json } | { ok: false; status: number; code: string }> {
  let response: Response;
  try {
    response = await fetch(gatewayCompletionsUrl(), {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${openAIKey}` },
      body: JSON.stringify(upstream),
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (err) {
    const name = err instanceof Error ? err.name : '';
    return { ok: false, status: 0, code: name === 'TimeoutError' ? 'timeout' : 'network_error' };
  }
  let parsed: unknown = null;
  try {
    parsed = await response.json();
  } catch {
    parsed = null;
  }
  if (!response.ok) {
    const error = isRecord(parsed) && isRecord(parsed.error) ? parsed.error : {};
    const code = typeof error.code === 'string' && error.code !== '' ? error.code : typeof error.type === 'string' ? error.type : `http_${response.status}`;
    return { ok: false, status: response.status, code };
  }
  if (!isRecord(parsed)) return { ok: false, status: response.status, code: 'bad_response' };
  return { ok: true, body: parsed };
}

/**
 * Serve one prompt: dial the best free model that meets its requirements, failing over down the
 * rating order when a model refuses or answers without the prompt's token. A model that fails is
 * cooled down for the rest of the trial. Returns the answer and the model that served it, or the
 * refusals when no model served the prompt (the 502 then still carries the trail header).
 */
async function servePrompt(api: 'openai' | 'anthropic', body: Json, row: PromptRow | undefined): Promise<ServedPrompt | UnservedPrompt> {
  const deadline = Date.now() + PROMPT_TIMEOUT_MS;
  const token = tokenOf(body, api);
  const requires = row !== undefined ? row.requires : requirementsFromBody(body, api);
  const trail: UpstreamFailure[] = [];
  const tried = new Set<string>();
  for (let attempt = 0; attempt < MAX_ATTEMPTS_PER_PROMPT; attempt++) {
    const remaining = deadline - Date.now();
    if (remaining <= 1000) break;
    const candidates = rankedCandidates(brief, requires, cooledDown).filter((m) => !tried.has(m.id));
    if (candidates.length === 0) break;
    const model = candidates[0];
    tried.add(model.id);
    const timeoutMs = Math.min(ATTEMPT_TIMEOUT_MS, remaining - 500);
    let upstream: Json;
    if (api === 'openai') {
      const { stream: _stream, stream_options: _streamOptions, ...rest } = body;
      upstream = { ...rest, model: model.id, stream: false, max_tokens: maxTokensOf(body) };
    } else {
      upstream = anthropicToOpenAI(body, model.id);
      upstream.max_tokens = maxTokensOf(body);
    }
    const result = await dial(model.id, upstream, timeoutMs);
    if (!result.ok) {
      console.error(`[aggregator] ${row?.id ?? '?'}: ${model.id} refused (${result.status} ${result.code})`);
      trail.push({ status: result.status, code: result.code, model: model.id });
      cooledDown.add(model.id);
      continue;
    }
    const text = completionText(result.body);
    if (text === '' || (token !== null && !text.includes(token))) {
      console.error(`[aggregator] ${row?.id ?? '?'}: ${model.id} answered without the token`);
      trail.push({ status: 200, code: 'bad_answer', model: model.id });
      cooledDown.add(model.id);
      continue;
    }
    console.error(`[aggregator] ${row?.id ?? '?'}: served by ${model.id} (${trail.length} refusal${trail.length === 1 ? '' : 's'} before it)`);
    return { answer: result.body, model: model.id, trail };
  }
  return { answer: null, trail };
}

/** The trail header's value: the refused attempts as JSON, at most eight rows. */
function trailHeader(trail: UpstreamFailure[]): string {
  return JSON.stringify(trail.slice(0, MAX_TRAIL_ROWS));
}

/** Send a JSON body, with the trail header when the prompt failed over. */
function sendJson(res: ServerResponse, status: number, body: Json, trail: UpstreamFailure[]): void {
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (trail.length > 0) headers['x-makefun-upstream-failures'] = trailHeader(trail);
  res.writeHead(status, headers);
  res.end(JSON.stringify(body));
}

/** Send a server-sent-event body, with the trail header when the prompt failed over. */
function sendSse(res: ServerResponse, body: string, trail: UpstreamFailure[]): void {
  const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
  if (trail.length > 0) headers['x-makefun-upstream-failures'] = trailHeader(trail);
  res.writeHead(200, headers);
  res.end(body);
}

/** Whether the request asks for a stream: `stream: true` on an OpenAI body, `stream: true` on an Anthropic one. */
function wantsStream(body: Json): boolean {
  return body.stream === true;
}

/** Serve one completion request on either wire. */
async function handleCompletion(api: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) {
    const error = api === 'anthropic' ? anthropicError('invalid_request_error', 'the request body is not a JSON object') : openAIError('the request body is not a JSON object', 'invalid_request_error', 'invalid_request');
    sendJson(res, 400, error, []);
    return;
  }
  const row = brief.prompts[promptIndex];
  promptIndex += 1;
  const served = await servePrompt(api, body, row);
  if (served.answer === null) {
    const error =
      api === 'anthropic'
        ? anthropicError('overloaded_error', 'no free model served this prompt')
        : openAIError('no free model served this prompt', 'server_error', 'no_model_served');
    sendJson(res, 502, error, served.trail);
    return;
  }
  if (api === 'openai') {
    if (wantsStream(body)) sendSse(res, openAIStreamFromAnswer(served.answer, served.model), served.trail);
    else sendJson(res, 200, openAIJsonWithModel(served.answer, served.model), served.trail);
  } else {
    if (wantsStream(body)) sendSse(res, anthropicStreamFromAnswer(served.answer, served.model), served.trail);
    else sendJson(res, 200, openAIToAnthropic(served.answer, served.model), served.trail);
  }
}

/** Which wire a model-list request is on: an x-api-key header names Anthropic, a Bearer token OpenAI. */
function modelListWire(req: IncomingMessage): 'openai' | 'anthropic' {
  const key = req.headers['x-api-key'];
  if (typeof key === 'string' && key !== '') return 'anthropic';
  return 'openai';
}

const server = createServer((req, res) => {
  const url = new URL(req.url ?? '/', 'http://localhost');
  const path = url.pathname;
  if (req.method === 'GET' && path === '/v1/models') {
    const body = modelListWire(req) === 'anthropic' ? anthropicModelList(servedModels) : openAIModelList(servedModels);
    sendJson(res, 200, body, []);
    return;
  }
  if (req.method === 'POST' && path === '/v1/chat/completions') {
    void handleCompletion('openai', req, res).catch((err: unknown) => {
      console.error(`[aggregator] chat/completions: ${(err as Error).message}`);
      if (!res.headersSent) sendJson(res, 500, openAIError('internal error', 'server_error', 'internal_error'), []);
    });
    return;
  }
  if (req.method === 'POST' && path === '/v1/messages') {
    void handleCompletion('anthropic', req, res).catch((err: unknown) => {
      console.error(`[aggregator] messages: ${(err as Error).message}`);
      if (!res.headersSent) sendJson(res, 500, anthropicError('api_error', 'internal error'), []);
    });
    return;
  }
  sendJson(res, 404, openAIError('not found', 'invalid_request_error', 'not_found'), []);
});

async function register(port: number): Promise<void> {
  const response = await fetch(`${envUrl.replace(/\/$/, '')}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: 1, port, catalog }),
    signal: AbortSignal.timeout(10_000),
  });
  if (!response.ok) throw new Error(`register failed: HTTP ${response.status}`);
}

async function pollWork(): Promise<boolean> {
  const response = await fetch(`${envUrl.replace(/\/$/, '')}/work`, { signal: AbortSignal.timeout(10_000) });
  if (!response.ok) return false;
  const parsed: unknown = await response.json();
  return isRecord(parsed) && parsed.done === true;
}

async function run(): Promise<void> {
  if (process.argv.includes('--list')) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }
  await new Promise<void>((resolve) => {
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      const port = typeof address === 'object' && address !== null ? address.port : 0;
      if (port === 8080) {
        // Port 8080 is the env's own: never serve on it. Re-listen on a fresh port.
        server.close(() => {
          server.listen(0, '0.0.0.0', () => resolve());
        });
        return;
      }
      resolve();
    });
  });
  const address = server.address();
  const port = typeof address === 'object' && address !== null ? address.port : 0;
  console.error(`[aggregator] listening on ${port} (${servedModels.length} free models)`);
  if (envUrl === '') {
    console.error('[aggregator] MAKEFUN_ENV_URL is not set: serving without registering');
    return;
  }
  await register(port);
  console.error('[aggregator] registered');
  const timer = setInterval(() => {
    void pollWork()
      .then((done) => {
        if (!done) return;
        clearInterval(timer);
        server.close(() => process.exit(0));
      })
      .catch((err: unknown) => console.error(`[aggregator] work poll: ${(err as Error).message}`));
  }, POLL_INTERVAL_MS);
  timer.unref();
}

process.on('SIGTERM', () => process.exit(0));
process.on('SIGINT', () => process.exit(0));

run().catch((err: unknown) => {
  console.error(`[aggregator] fatal: ${(err as Error).message}`);
  process.exit(1);
});
