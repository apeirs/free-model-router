// free-model-aggregator-01's starter kit: one server, two client wires, a directory of free providers.
//
// This is a working entry to start from, not a competitive answer. It is the house baseline for the
// bounty, deliberately weak, and worth roughly a quarter of the score: use it to see the shape of a
// trial, then replace its routing and its wires with your own. The gaps are marked TODO below.
//
// WHAT A TRIAL GIVES YOU
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from your
//   repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
//   directly, and a relative import needs its .ts extension). The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//     - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//       anthropic or gemini), auth scheme, key environment variable, the key the service takes, and the
//       base path you join to $MAKEFUN_ENV_URL;
//     - the quality table: every model's public rating and what it can do (tools, vision, JSON mode,
//       context length), and whether it is free;
//     - the workload manifest: the twenty requests the client will send, in order, with each one's api,
//       whether it streams, its weight and the requirements a model must meet to serve it.
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads
//   the brief, probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//        x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//        /v1/messages, each with a JSON body or with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
//   body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// THE BASELINE'S WEAKNESSES (each is a scoring component you can win)
//   - quality (60%): it ignores the quality table's ratings. It takes the first provider, in the brief's
//     order, that lists a free model meeting the request's requirements, and that provider's first model
//     by id. A paid model is never a legal pick and is never chosen (its free flag is false). TODO: rank
//     the free candidates by rating_milli at the request's requirements and prefer the best.
//   - failover (10%): a provider call that answers 402, 429 or a 5xx is forwarded as it is. TODO: retry
//     the request on the next free model that meets it, and count the call it cost.
//   - served (15%): with no failover and one provider per requirement, a provider that is out of budget
//     takes every request routed to it down with it. TODO: spread calls over providers under their
//     per-trial budgets.
//   - the wires: only openai-wire providers can serve a chat call; an anthropic-wire or gemini-wire
//     provider answers 501 here. The catalog probe already speaks all three wires. TODO: build the other
//     two upstream requests and their event shapes.
//   - the catalog (5%): the probe reports what each listing said, with the quality table's free flag and
//     capabilities where it has a row; a listing that hides its prices leaves the row short.
//   The trial's rules are in the kit's RULES.txt; read them for the score's columns and the tie-break.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        Anthropic <-> Chat Completions translation and the SSE writers.
//   LICENSE            MIT: keep it or replace it with another OSI-approved license.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import { anthropicError, anthropicModelList, openAIError, openAIModelList, type ServedModel } from './lib/wire.ts';
import {
  anthropicJson,
  anthropicSse,
  canonical,
  fromAnthropic,
  fromGemini,
  fromOpenAI,
  openAIJson,
  openAISse,
  toAnthropic,
  toGemini,
  toOpenAI,
  wantsJson,
  type Result,
} from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** The most calls each provider can take in a trial (the top of its drawn budget): a call past it is refused. */
const CALLS_CEILING: Record<string, number> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};
/** How many times one provider is tried for one request before it is given up for the trial. */
const ATTEMPTS = 3;
/** The longest a retry-after is waited out, in milliseconds. */
const MAX_RETRY_WAIT_MS = 1500;
/** How long one provider call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 20000;

/** Providers given up for the trial: refused every attempt, or past their ceiling. */
const dead = new Set<string>();
/** Calls this server has made to each provider that counted against its budget. */
const callsMade = new Map<string, number>();

/**
 * Every free model meeting the requirements, best first: the quality table's rating, then the provider's
 * place in the brief, then the model id. A provider on the anthropic wire cannot carry JSON mode.
 */
function candidatesFor(requires: Requirement, json: boolean): Route[] {
  const out: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok || dead.has(probe.provider.id)) continue;
    if (json && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  const order = (r: Route): number => probes.indexOf(r.probe);
  return out.sort((a, b) => b.rating - a.rating || order(a) - order(b) || byId(a.model, b.model));
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(api: 'openai' | 'anthropic', body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined && entry.api === api) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One upstream call's outcome: an answer, a refusal worth retrying, or a request this model cannot take. */
type Outcome = { kind: 'ok'; result: Result } | { kind: 'refused'; waitMs: number } | { kind: 'rejected' };

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/** Call one model once, in its provider's wire, without streaming. */
async function callOnce(route: Route, canon: Json, anthropicVersion: string): Promise<Outcome> {
  const provider = route.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let body: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = toOpenAI(canon, route.model.id);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = anthropicVersion;
    body = toAnthropic(canon, route.model.id);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(route.model.id)}:generateContent`;
    body = toGemini(canon);
  }
  let status = 0;
  let text = '';
  let retryAfter = '';
  try {
    const upstream = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
    status = upstream.status;
    retryAfter = upstream.headers.get('retry-after') ?? '';
    text = await upstream.text();
  } catch (err) {
    log(`${provider.id}/${route.model.id}: ${(err as Error).message}`);
    return { kind: 'refused', waitMs: 0 };
  }
  if (status !== 400 && status !== 401) callsMade.set(provider.id, (callsMade.get(provider.id) ?? 0) + 1);
  if (status === 402 || status === 429 || status >= 500) {
    const seconds = Number(retryAfter);
    return { kind: 'refused', waitMs: Number.isFinite(seconds) && seconds > 0 ? Math.min(seconds * 1000, MAX_RETRY_WAIT_MS) : 0 };
  }
  if (status < 200 || status >= 300) {
    log(`${provider.id}/${route.model.id}: HTTP ${status} ${text.slice(0, 200)}`);
    return { kind: 'rejected' };
  }
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch {
    return { kind: 'rejected' };
  }
  if (!isRecord(json)) return { kind: 'rejected' };
  const result = provider.wire === 'openai' ? fromOpenAI(json, route.model.id) : provider.wire === 'anthropic' ? fromAnthropic(json, route.model.id) : fromGemini(json, route.model.id);
  return result === null ? { kind: 'rejected' } : { kind: 'ok', result };
}

/**
 * Answer one request with the best free model that takes it. A refusal (402, 429, 5xx) is retried on the
 * same provider, waiting out a retry-after, since a scheduled refusal clears on the next call; a provider
 * that refuses every attempt, or has been called past its ceiling, is given up for the trial.
 */
async function answer(canon: Json, requires: Requirement, anthropicVersion: string): Promise<Result | null> {
  const tried = new Set<string>();
  for (;;) {
    const route = candidatesFor(requires, wantsJson(canon)).find((r) => !tried.has(`${r.probe.provider.id}/${r.model.id}`));
    if (route === undefined) return null;
    const id = route.probe.provider.id;
    tried.add(`${id}/${route.model.id}`);
    for (let attempt = 1; attempt <= ATTEMPTS; attempt += 1) {
      const outcome = await callOnce(route, canon, anthropicVersion);
      if (outcome.kind === 'ok') return outcome.result;
      if (outcome.kind === 'rejected') break;
      if ((callsMade.get(id) ?? 0) >= (CALLS_CEILING[id] ?? Infinity) || attempt === ATTEMPTS) {
        dead.add(id);
        break;
      }
      if (outcome.waitMs > 0) await sleep(outcome.waitMs);
    }
  }
}

/** POST /v1/chat/completions or /v1/messages: read the client's wire, answer from the best free model, answer in the client's wire. */
async function serve(api: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, api, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const canon = canonical(api, body);
  const version = typeof req.headers['anthropic-version'] === 'string' ? req.headers['anthropic-version'] : '2023-06-01';
  const result = await answer(canon, requirementsFor(api, body), version);
  if (result === null) {
    return sendError(res, api, 503, api === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model could answer this request');
  }
  if (body.stream === true) {
    const text = api === 'openai' ? openAISse(result) : anthropicSse(result);
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(text);
    return;
  }
  sendJson(res, 200, api === 'openai' ? openAIJson(result) : anthropicJson(result));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
