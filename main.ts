// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// This server reads the trial brief, probes each provider's catalog route in the
// provider's own wire, and serves the OpenAI Chat Completions API and the
// Anthropic Messages API over the best free model that can serve each request:
// in JSON and in server-sent events, on any of the three upstream wires.
//
// HOW A REQUEST IS SERVED
//   The manifest's row for the request (in arrival order) names what the request
//   needs; a body with no manifest row is read off its own fields instead. The
//   candidates are every free model meeting those requirements, best first by
//   the quality table's rating, then by the fewest refusals, then the fewest
//   calls, then the model id. A JSON-mode request never goes to an
//   anthropic-wire provider: that wire cannot carry JSON mode. The server calls
//   the best candidate; a refusal (402, 429, 5xx, a timeout, a body that cannot
//   be read, or a 2xx that carries no usable text) is counted
//   against that provider and the request moves to the next candidate, one
//   provider per attempt and at most eight attempts. A 402, a spent-budget or
//   outage message, or a third refusal spends the provider for the rest of the
//   trial. A 2xx answer that carries usable text is never retried: its text and usage are returned in
//   the wire the client used, whole and exactly as the provider reported them.
//
// HOW A TRIAL RUNS
//   The server listens on 0.0.0.0 at a free port, registers it with the env
//   (POST $MAKEFUN_ENV_URL/register, protocol version 1, with the one catalog),
//   serves the client's twenty requests, polls GET $MAKEFUN_ENV_URL/work every
//   100 ms and exits 0 the moment the env says the workload is done. A process
//   still running when the wall ends scores zero, so nothing else is waited for.
//
// `node main.ts --list` prints the same catalog JSON the register body carries
// and exits 0 before any server starts. Run outside a trial, with no
// $MAKEFUN_PROMPT_FILE and no $MAKEFUN_ENV_URL set, it prints {"providers":[]}
// and exits 0: it has no brief to read and no provider route to probe.
//
// Every choice is a function of the brief, the requests and the providers'
// answers: no wall clock and no unseeded randomness, so a replay leaves the
// same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  candidatesFor,
  isRecord,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  RouterBooks,
  type Brief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  errorMessageOf,
  openAIError,
  openAIModelList,
  parseUpstreamAnswer,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
  upstreamCall,
  type NormalizedAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 15000;
/** The most attempts one request gets, one provider per attempt. */
const MAX_ATTEMPTS = 8;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
const books = new RouterBooks();
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

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
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

/** Whether this request asks for JSON mode: the manifest's requirement, or the client's own response_format. */
function jsonModeOf(body: Json, requires: Requirement): boolean {
  if (requires.json === true) return true;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
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

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
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

interface ServedAnswer {
  answer: NormalizedAnswer;
  model: string;
}

/**
 * Serve one request on the best free model that can serve it. Each attempt calls
 * the best candidate not yet tried for this request; a refusal is counted against
 * its provider and the next candidate is tried, up to eight attempts. A 2xx
 * answer that carries usable text ends the loop at once and is never retried.
 * A 2xx whose body cannot be read, or that carries no text at all, counts as a
 * refusal instead: the client checks the provider's text whole, so an empty
 * answer earns the request nothing. Null when no candidate answered.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, requires: Requirement): Promise<ServedAnswer | null> {
  const jsonMode = jsonModeOf(body, requires);
  const stream = body.stream === true;
  const tried = new Set<string>();
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    const candidates = candidatesFor(probes, brief.ratings, books, requires, tried);
    if (candidates.length === 0) return null;
    const candidate = candidates[0];
    const providerId = candidate.probe.provider.id;
    tried.add(providerId);
    const call = upstreamCall(ENV_URL, candidate.probe.provider, candidate.model.id, clientApi, body, jsonMode);
    let upstream: Response;
    try {
      upstream = await fetch(call.url, {
        method: 'POST',
        headers: call.headers,
        body: call.body,
        signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
      });
    } catch (err) {
      books.noteRefusal(providerId, 0, (err as Error).message);
      log(`${providerId}/${candidate.model.id} unreachable: ${(err as Error).message}`);
      continue;
    }
    const text = await upstream.text().catch((err: unknown) => {
      // The provider answered 2xx but its body could not be read (the connection
      // dropped mid-body, a truncated response): that is a refusal like any other,
      // so the request moves to the next candidate instead of failing the client.
      books.noteRefusal(providerId, upstream.status, `the provider's answer could not be read: ${(err as Error).message}`);
      log(`${providerId}/${candidate.model.id} answer unreadable: ${(err as Error).message}`);
      return null;
    });
    if (text === null) continue;
    if (!upstream.ok) {
      const message = errorMessageOf(candidate.probe.provider.wire, text);
      books.noteRefusal(providerId, upstream.status, message);
      log(`${providerId}/${candidate.model.id} refused HTTP ${upstream.status}: ${message.slice(0, 200)}`);
      continue;
    }
    books.noteCall(providerId);
    const answer = parseUpstreamAnswer(candidate.probe.provider.wire, stream, text);
    if (answer === null) {
      books.noteRefusal(providerId, upstream.status, 'the provider answered no text');
      log(`${providerId}/${candidate.model.id} answered HTTP ${upstream.status} with no usable answer`);
      continue;
    }
    if (answer.text === '') {
      // A 2xx that carries no text serves nothing: the client checks the
      // provider's text whole, so an empty answer earns that request nothing.
      // Treat it like a refusal and let the next candidate answer instead.
      books.noteRefusal(providerId, upstream.status, 'the provider answered an empty text');
      log(`${providerId}/${candidate.model.id} answered HTTP ${upstream.status} with an empty text`);
      continue;
    }
    return { answer, model: candidate.model.id };
  }
  return null;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered on the best free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const served = await serve('openai', body, requires);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
  if (body.stream === true) {
    sendText(res, 200, 'text/event-stream; charset=utf-8', renderOpenAIStream(served.answer, served.model));
    return;
  }
  sendJson(res, 200, renderOpenAIJson(served.answer, served.model));
}

/** POST /v1/messages: an Anthropic-shaped client, answered on the best free model. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const served = await serve('anthropic', body, requires);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
  if (body.stream === true) {
    sendText(res, 200, 'text/event-stream; charset=utf-8', renderAnthropicStream(served.answer, served.model));
    return;
  }
  sendJson(res, 200, renderAnthropicJson(served.answer, served.model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
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
