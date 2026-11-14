// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the trial brief (provider directory, sealed quality table,
// workload manifest), probes every provider's catalog route in its own wire,
// registers one catalog with the env, then answers the env client's 20
// requests, each with the highest-rated free model that meets the request's
// requirements, failing over to the next-best free model whenever a provider
// refuses (402/429/5xx). It exits 0 when GET /work says the workload is done.
//
// Routing is a pure function of the brief and the provider answers seen so
// far: candidates are ordered by the quality table's rating_milli (ties break
// toward the provider this server has called least, then by provider id, then
// model id), so a live pass and its replay make the same picks. No wall clock
// and no unseeded randomness anywhere: fixed timestamps in every listing and
// answer, no Date.now(), no Math.random().
//
// The three upstream wires are spoken natively, because the echo check
// compares what the provider parsed against what the client sent: a field
// rewritten or dropped loses the request.
//   - OpenAI client -> OpenAI provider: passthrough (model swapped, and
//     stream_options.include_usage added on streams so the usage chunk arrives).
//   - OpenAI client -> Anthropic provider: system/user/assistant/tool messages,
//     tools, tool_choice, stops and max_tokens rewritten into Messages shape.
//   - OpenAI client -> Gemini provider: generateContent with systemInstruction,
//     contents, functionDeclarations and generationConfig.
//   - Anthropic client -> OpenAI provider: the mirror translation.
//   - Anthropic client -> Anthropic provider: passthrough.
//   - Anthropic client -> Gemini provider: Messages shape into generateContent.
// JSON-mode requests from an OpenAI client are never routed to an Anthropic
// wire (it cannot carry response_format, so the provider would echo json_mode
// false and the echo check would fail); they go to an OpenAI or Gemini wire.
// A provider's answer is returned whole in the wire the client used: its
// text verbatim, its usage counts as that wire reports them, its stop reason
// mapped stop/end_turn/STOP -> stop/end_turn. Providers only ever answer
// family-0 stop reasons, so the client always sees stop / end_turn.

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNumber = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;
const asBool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Fixed epoch second stamped on every OpenAI listing and answer: never the clock. */
const FIXED_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
const ANTHROPIC_VERSION = '2023-06-01';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: provider directory, quality table, workload manifest.
// ---------------------------------------------------------------------------

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  workload: WorkloadRequest[];
  error: string | null;
}

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** The index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` carrying a string `schema`, in order of appearance. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none'
      ? v.auth
      : 'bearer';
  const base =
    asString(v.path) !== ''
      ? asString(v.path)
      : asString(v.base_path) !== ''
        ? asString(v.base_path)
        : asString(v.base_url) !== ''
          ? asString(v.base_url)
          : asString(v.base);
  return {
    id,
    name: asString(v.name) !== '' ? asString(v.name) : id,
    wire,
    auth,
    base,
    key_env: asString(v.key_env),
    key: asString(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: asNumber(v.rating_milli) ?? 0,
    context_length: asNumber(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  const req: Requirement = {
    tools: requires.tools === true,
    vision: requires.vision === true,
    json: requires.json === true,
  };
  if (typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)) {
    req.min_context = requires.min_context;
  }
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: req,
  };
}

function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null || !Array.isArray(dirDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)
      ? ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        })
      : [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload =
    workloadDoc !== null && Array.isArray(workloadDoc.manifest)
      ? workloadDoc.manifest.flatMap((v) => {
          const w = workloadOf(v);
          return w === null ? [] : [w];
        })
      : [];
  return { providers, ratings, workload, error: null };
}

// ---------------------------------------------------------------------------
// Catalog probe: each provider's own catalog route, in its own wire.
// ---------------------------------------------------------------------------

interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface CatalogReport {
  providers: CatalogProvider[];
}

function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** A listing's price field: true when every price it names is zero, null when it says nothing. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? asString(v.name).replace(/^models\//, '') : asString(v.id);
  if (id === '') return null;
  return {
    id,
    display_name:
      asString(v.display_name) !== ''
        ? asString(v.display_name)
        : asString(v.displayName) !== ''
          ? asString(v.displayName)
          : id,
    free: asBool(v.free) ?? zeroPriced(v),
    context_length: asNumber(v.context_length) ?? asNumber(v.context_window) ?? asNumber(v.inputTokenLimit),
    tools: asBool(v.tools),
    vision: asBool(v.vision),
    json: asBool(v.json),
  };
}

function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs = PROBE_TIMEOUT_MS): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = {
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) {
      return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    }
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

function ratingRowOf(ratings: RatingRow[], providerId: string, modelId: string): RatingRow | null {
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) {
    return false;
  }
  return true;
}

/**
 * The one catalog: every provider that answered, in the brief's order, every
 * model it listed, sorted by id. base_url is the directory's own path
 * (/p/<id>), never joined to the env origin. The same object is printed by
 * --list and posted to /register, so the two agree byte for byte.
 */
function buildCatalog(probes: readonly Probe[], ratings: RatingRow[]): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRowOf(ratings, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: modelFree(model, row),
        context_length: row?.context_length ?? model.context_length ?? null,
      });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Routing: highest-rated free model meeting the requirements, with failover.
// ---------------------------------------------------------------------------

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** Calls this server has made per provider, used only to break rating ties. */
const callsMade = new Map<string, number>();
/** Requests served so far, in arrival order: the manifest is the source of requirements. */
let manifestIndex = 0;

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  row: RatingRow | null;
  rating: number;
}

/** Free candidates meeting the requirements, best first. */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingRowOf(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      // An Anthropic-wire provider cannot carry response_format: a JSON-mode
      // request from an OpenAI client must go to a wire that can.
      if (requires.json === true && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, row, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const ca = callsMade.get(a.probe.provider.id) ?? 0;
    const cb = callsMade.get(b.probe.provider.id) ?? 0;
    if (ca !== cb) return ca - cb;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** Requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, clientApi: 'openai' | 'anthropic'): Requirement {
  const manifest = brief.workload;
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body, clientApi);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json, clientApi: 'openai' | 'anthropic'): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (clientApi === 'openai' && (block.type === 'image' || block.type === 'image_url')) vision = true;
      if (clientApi === 'anthropic' && block.type === 'image') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

// ---------------------------------------------------------------------------
// Request translation: client wire -> provider wire.
// The echo check compares what the provider parsed against what the client
// sent, so every translated field must carry the same text, tool names,
// max_tokens, stops, image bytes and JSON-mode flag.
// ---------------------------------------------------------------------------

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** OpenAI image_url part -> Anthropic image block. */
function openAIImageToAnthropic(part: Json): Json | null {
  if (!isRecord(part.image_url)) return null;
  const url = asString(part.image_url.url);
  if (url.startsWith('data:')) {
    const m = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (m === null) return null;
    return {
      type: 'image',
      source: { type: 'base64', media_type: m[1] !== '' ? m[1] : 'image/png', data: m[2] },
    };
  }
  return { type: 'image', source: { type: 'url', url } };
}

/** Anthropic image block -> OpenAI image_url part. */
function anthropicImageToOpenAI(block: Json): Json | null {
  if (!isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** An image in either client wire -> a Gemini part (inlineData for base64, fileData for a URL). */
function imageToGeminiPart(url: string, mime: string): Json | null {
  if (url.startsWith('data:')) {
    const m = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (m === null) return null;
    return { inlineData: { mimeType: m[1] !== '' ? m[1] : mime, data: m[2] } };
  }
  if (url !== '') return { fileData: { mimeType: mime, fileUri: url } };
  return null;
}

function openAIImageToGemini(part: Json): Json | null {
  if (!isRecord(part.image_url)) return null;
  return imageToGeminiPart(asString(part.image_url.url), 'image/png');
}

function anthropicImageToGemini(block: Json): Json | null {
  if (!isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: asString(source.media_type) !== '' ? asString(source.media_type) : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { fileData: { mimeType: 'image/png', fileUri: source.url } };
  }
  return null;
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An Anthropic Messages request as an OpenAI Chat Completions request naming `model`. */
function anthropicToOpenAI(body: Json, model: string): Json {
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
          id: asString(block.id),
          type: 'function',
          function: {
            name: asString(block.name),
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
          tool_call_id: asString(block.tool_use_id),
          content: toolResultText(block.content),
        });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
      if (block.type === 'image') {
        const part = anthropicImageToOpenAI(block);
        if (part !== null) parts.push(part);
      }
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
            description: asString(raw.description),
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
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** An OpenAI Chat Completions request as an Anthropic Messages request naming `model`. */
function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = asString(raw.role);
    const content = raw.content;
    if (role === 'system') {
      const text = textOf(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: Json = {};
        const args = call.function.arguments;
        if (isRecord(args)) input = args;
        else if (typeof args === 'string' && args.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(args);
            if (isRecord(parsed)) input = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        }
        blocks.push({
          type: 'tool_use',
          id: asString(call.id) !== '' ? asString(call.id) : 'toolu_unknown',
          name: asString(call.function.name),
          input,
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: asString(raw.tool_call_id), content: toolResultText(content) }],
      });
      continue;
    }
    // user
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (Array.isArray(content)) {
      const blocks: Json[] = [];
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
        if (part.type === 'image_url') {
          const block = openAIImageToAnthropic(part);
          if (block !== null) blocks.push(block);
        }
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stringList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: asString(raw.function.description),
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: choice === 'required' ? 'any' : 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function)) {
    out.tool_choice = { type: 'tool', name: asString(choice.function.name) };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** OpenAI tools as Gemini functionDeclarations, names and order kept. */
function openAIToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: asString(raw.function.description),
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

/** Anthropic tools as Gemini functionDeclarations, names and order kept. */
function anthropicToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: asString(raw.description),
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
}

/** An OpenAI Chat Completions request as a Gemini generateContent body. */
function openAIToGemini(body: Json, stream: boolean): Json {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = asString(raw.role);
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let args: Json = {};
        const a = call.function.arguments;
        if (isRecord(a)) args = a;
        else if (typeof a === 'string' && a.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(a);
            if (isRecord(parsed)) args = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        }
        parts.push({ functionCall: { name: asString(call.function.name), args } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      contents.push({
        role: 'user',
        parts: [{ functionResponse: { name: '', response: { result: toolResultText(raw.content) } } }],
      });
      continue;
    }
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        if (part.type === 'image_url') {
          const p = openAIImageToGemini(part);
          if (p !== null) parts.push(p);
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts.map((t) => ({ text: t })) };
  const declarations = openAIToolsToGemini(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') config.maxOutputTokens = body.max_completion_tokens;
  const stops = stringList(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  void stream;
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent body. */
function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (assistant) {
      const parts: Json[] = [];
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          if (block.type === 'tool_use') {
            parts.push({
              functionCall: { name: asString(block.name), args: isRecord(block.input) ? block.input : {} },
            });
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        if (block.type === 'image') {
          const p = anthropicImageToGemini(block);
          if (p !== null) parts.push(p);
        }
        if (block.type === 'tool_result') {
          parts.push({
            functionResponse: { name: '', response: { result: toolResultText(block.content) } },
          });
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = anthropicToolsToGemini(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  const stops = stringList(body.stop_sequences);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// One provider call: build the upstream request for the candidate's wire.
// ---------------------------------------------------------------------------

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

function upstreamCall(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
): UpstreamCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  const stream = body.stream === true;
  if (provider.wire === 'openai') {
    if (stream) headers.accept = 'text/event-stream';
    const upstream = clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream && !isRecord(upstream.stream_options)) upstream.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, headers, body: upstream };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = ANTHROPIC_VERSION;
    if (stream) headers.accept = 'text/event-stream';
    const upstream = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropic(body, candidate.model.id);
    return { url: `${base}/v1/messages`, headers, body: upstream };
  }
  // gemini
  const encoded = encodeURIComponent(candidate.model.id);
  const route = stream ? ':streamGenerateContent' : ':generateContent';
  const query = stream ? '?alt=sse' : '';
  const upstream = clientApi === 'openai' ? openAIToGemini(body, stream) : anthropicToGemini(body);
  return { url: `${base}/v1beta/models/${encoded}${route}${query}`, headers, body: upstream };
}

/** A provider call that refuses (402/429/5xx) or cannot be reached: try the next candidate. */
function isRetryable(status: number): boolean {
  return status === 402 || status === 429 || status >= 500;
}

// ---------------------------------------------------------------------------
// Response conversion: provider wire -> client wire.
// A provider's answer is returned whole: its text verbatim, its usage counts
// as that wire reports them, its stop reason mapped to the same family.
// ---------------------------------------------------------------------------

interface ToolCall {
  id: string;
  name: string;
  args: string;
}

interface NormalizedAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** Stop family: 0 stop, 1 length, 2 tool call, 3 refusal. Providers only ever answer 0. */
  stopFamily: number;
  toolCalls: ToolCall[];
}

const EMPTY_ANSWER: NormalizedAnswer = { text: '', inputTokens: 0, outputTokens: 0, stopFamily: 0, toolCalls: [] };

function stopFamilyOf(reason: unknown): number {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return 0;
}

function openAIStop(family: number): string {
  return ['stop', 'length', 'tool_calls', 'content_filter'][family] ?? 'stop';
}

function anthropicStop(family: number): string {
  return ['end_turn', 'max_tokens', 'tool_use', 'refusal'][family] ?? 'end_turn';
}

function numberOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function argumentsObject(value: unknown): Json {
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

/** One SSE frame: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

interface SseFrame {
  event: string | null;
  data: string;
}

/** Split an SSE body into frames (event name + data lines joined with \n). */
function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function splitText(text: string, size = 13): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += size) pieces.push(text.slice(at, at + size));
  return pieces.length === 0 ? [''] : pieces;
}

function openAIJsonAnswer(body: Json, model: string): NormalizedAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = isRecord(body.usage) ? body.usage : {};
  const toolCalls: ToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      toolCalls.push({
        id: asString(call.id),
        name: asString(call.function.name),
        args: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
      });
    }
  }
  return {
    text: typeof message.content === 'string' ? message.content : textOf(message.content),
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    stopFamily: stopFamilyOf(choice.finish_reason),
    toolCalls,
  };
}

function anthropicJsonAnswer(body: Json): NormalizedAnswer {
  const content = Array.isArray(body.content) ? body.content : [];
  const usage = isRecord(body.usage) ? body.usage : {};
  const textParts: string[] = [];
  const toolCalls: ToolCall[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') textParts.push(block.text);
    if (block.type === 'tool_use') {
      toolCalls.push({
        id: asString(block.id),
        name: asString(block.name),
        args: JSON.stringify(isRecord(block.input) ? block.input : {}),
      });
    }
  }
  return {
    text: textParts.join(''),
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    stopFamily: stopFamilyOf(body.stop_reason),
    toolCalls,
  };
}

function geminiJsonAnswer(body: Json): NormalizedAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []));
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: texts.join(''),
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    stopFamily: stopFamilyOf(candidate.finishReason),
    toolCalls: [],
  };
}

function openAIStreamAnswer(text: string): NormalizedAnswer {
  const out: NormalizedAnswer = { text: '', inputTokens: 0, outputTokens: 0, stopFamily: 0, toolCalls: [] };
  const byIndex = new Map<number, ToolCall>();
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      out.inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      out.outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length === 0) continue;
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out.text += delta.content;
    if (Array.isArray(delta.tool_calls)) {
      for (const call of delta.tool_calls) {
        if (!isRecord(call)) continue;
        const index = typeof call.index === 'number' ? call.index : byIndex.size;
        let entry = byIndex.get(index);
        if (entry === undefined) {
          entry = { id: '', name: '', args: '' };
          byIndex.set(index, entry);
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof call.id === 'string' && call.id !== '') entry.id = call.id;
        if (typeof fn.name === 'string' && fn.name !== '') entry.name = fn.name;
        if (typeof fn.arguments === 'string') entry.args += fn.arguments;
      }
    }
    if (typeof choice.finish_reason === 'string') out.stopFamily = stopFamilyOf(choice.finish_reason);
  }
  out.toolCalls = [...byIndex.values()];
  return out;
}

function anthropicStreamAnswer(text: string): NormalizedAnswer {
  const out: NormalizedAnswer = { text: '', inputTokens: 0, outputTokens: 0, stopFamily: 0, toolCalls: [] };
  const byIndex = new Map<number, ToolCall>();
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    switch (frame.event) {
      case 'message_start': {
        if (isRecord(data.message) && isRecord(data.message.usage)) {
          out.inputTokens = numberOrZero(data.message.usage.input_tokens);
        }
        break;
      }
      case 'content_block_delta': {
        const index = typeof data.index === 'number' ? data.index : -1;
        const delta = isRecord(data.delta) ? data.delta : {};
        if (delta.type === 'text_delta' && typeof delta.text === 'string') out.text += delta.text;
        if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && index >= 0) {
          let entry = byIndex.get(index);
          if (entry === undefined) {
            entry = { id: '', name: '', args: '' };
            byIndex.set(index, entry);
          }
          entry.args += delta.partial_json;
        }
        break;
      }
      case 'content_block_start': {
        const index = typeof data.index === 'number' ? data.index : -1;
        const block = isRecord(data.content_block) ? data.content_block : {};
        if (block.type === 'tool_use' && index >= 0) {
          byIndex.set(index, {
            id: asString(block.id),
            name: asString(block.name),
            args: '',
          });
        }
        break;
      }
      case 'message_delta': {
        const delta = isRecord(data.delta) ? data.delta : {};
        if (typeof delta.stop_reason === 'string') out.stopFamily = stopFamilyOf(delta.stop_reason);
        if (isRecord(data.usage)) out.outputTokens = numberOrZero(data.usage.output_tokens);
        break;
      }
      default:
        break;
    }
  }
  out.toolCalls = [...byIndex.values()];
  return out;
}

function geminiStreamAnswer(text: string): NormalizedAnswer {
  const out: NormalizedAnswer = { text: '', inputTokens: 0, outputTokens: 0, stopFamily: 0, toolCalls: [] };
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === 'string') out.text += part.text;
    }
    if (typeof candidate.finishReason === 'string') out.stopFamily = stopFamilyOf(candidate.finishReason);
    if (isRecord(data.usageMetadata)) {
      out.inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      out.outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  return out;
}

/** A normalized answer as an OpenAI Chat Completions JSON object. */
function toOpenAIJson(answer: NormalizedAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({
      id: c.id !== '' ? c.id : 'call_unknown',
      type: 'function',
      function: { name: c.name, arguments: c.args },
    }));
  }
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIStop(answer.stopFamily), logprobs: null }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** A normalized answer as an Anthropic Messages JSON object. */
function toAnthropicJson(answer: NormalizedAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({
      type: 'tool_use',
      id: call.id !== '' ? call.id : 'toolu_unknown',
      name: call.name,
      input: argumentsObject(call.args),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStop(answer.stopFamily),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A normalized answer as an OpenAI SSE stream (chunks + usage chunk + [DONE]). */
function toOpenAIStream(answer: NormalizedAnswer, model: string): string {
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent(
        null,
        chunk({ tool_calls: [{ index: 0, id: call.id, type: 'function', function: { name: call.name, arguments: call.args } }] }, null),
      ),
    );
  }
  frames.push(sseEvent(null, chunk({}, openAIStop(answer.stopFamily))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregator',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalized answer as an Anthropic SSE stream (message_start ... message_stop). */
function toAnthropicStream(answer: NormalizedAnswer, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  );
  let nextIndex = 0;
  const textIndex = nextIndex;
  nextIndex += 1;
  frames.push(
    sseEvent('content_block_start', {
      type: 'content_block_start',
      index: textIndex,
      content_block: { type: 'text', text: '' },
    }),
  );
  for (const piece of splitText(answer.text)) {
    frames.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: textIndex,
        delta: { type: 'text_delta', text: piece },
      }),
    );
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
  for (const call of answer.toolCalls) {
    const index = nextIndex;
    nextIndex += 1;
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: {
          type: 'tool_use',
          id: call.id !== '' ? call.id : `toolu_${index}`,
          name: call.name,
          input: {},
        },
      }),
    );
    if (call.args !== '') {
      frames.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: call.args },
        }),
      );
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStop(answer.stopFamily), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
  );
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

/** Normalize one upstream answer (JSON or SSE) into the wire-neutral form. */
function normalizeUpstream(wire: Wire, stream: boolean, bodyText: string): NormalizedAnswer {
  if (stream) {
    if (wire === 'openai') return openAIStreamAnswer(bodyText);
    if (wire === 'anthropic') return anthropicStreamAnswer(bodyText);
    return geminiStreamAnswer(bodyText);
  }
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(bodyText);
  } catch {
    return EMPTY_ANSWER;
  }
  if (!isRecord(parsed)) return EMPTY_ANSWER;
  if (wire === 'openai') return openAIJsonAnswer(parsed, '');
  if (wire === 'anthropic') return anthropicJsonAnswer(parsed);
  return geminiJsonAnswer(parsed);
}

/** Whether two wires match closely enough to relay the bytes untouched. */
function sameWire(a: Wire, b: 'openai' | 'anthropic'): boolean {
  return a === b;
}

// ---------------------------------------------------------------------------
// The server: GET /v1/models, POST /v1/chat/completions, POST /v1/messages.
// ---------------------------------------------------------------------------

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

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

/** The free models this server lists, deduplicated by id, in catalog order. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingRowOf(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
    });
  }
}

/**
 * Serve one client request: try the ranked candidates in order until one
 * answers 2xx, then return its answer in the client's wire. A refusal
 * (402/429/5xx) or an unreachable provider moves to the next candidate; a
 * 2xx is always the answer, even if converting it is best-effort, so exactly
 * one provider call ever answers a request.
 */
async function serve(
  req: IncomingMessage,
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
  body: Json,
): Promise<void> {
  const requires = requirementsFor(body, clientApi);
  const candidates = candidatesFor(requires, clientApi);
  const stream = body.stream === true;
  for (const candidate of candidates) {
    const call = upstreamCall(candidate, clientApi, body);
    let upstream: Response;
    try {
      upstream = await fetch(call.url, {
        method: 'POST',
        headers: call.headers,
        body: JSON.stringify(call.body),
        signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
      });
    } catch {
      continue;
    }
    callsMade.set(candidate.probe.provider.id, (callsMade.get(candidate.probe.provider.id) ?? 0) + 1);
    if (!upstream.ok) {
      if (isRetryable(upstream.status)) {
        try {
          await upstream.arrayBuffer();
        } catch {
          // Drained or not: move on.
        }
        continue;
      }
      // A 4xx that is not a refusal (bad model, bad key) is not retryable on
      // this candidate; another candidate may still serve it.
      try {
        await upstream.arrayBuffer();
      } catch {
        // Drained or not: move on.
      }
      continue;
    }
    const modelId = candidate.model.id;
    if (sameWire(candidate.probe.provider.wire, clientApi)) {
      // Same wire both sides: relay the provider's bytes untouched.
      const contentType = upstream.headers.get('content-type') ?? 'application/json';
      res.writeHead(200, { 'content-type': contentType });
      if (upstream.body === null) {
        res.end();
        return;
      }
      const reader = upstream.body.getReader();
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          res.write(Buffer.from(value));
        }
      } finally {
        reader.releaseLock();
      }
      res.end();
      return;
    }
    // Cross-wire: read the whole answer, normalize it, re-emit it.
    const bodyText = await upstream.text();
    const answer = normalizeUpstream(candidate.probe.provider.wire, stream, bodyText);
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.end(clientApi === 'openai' ? toOpenAIStream(answer, modelId) : toAnthropicStream(answer, modelId));
      return;
    }
    sendJson(res, 200, clientApi === 'openai' ? toOpenAIJson(answer, modelId) : toAnthropicJson(answer, modelId));
    return;
  }
  // No candidate answered: nothing legal could serve this request.
  sendError(
    res,
    clientApi,
    503,
    clientApi === 'anthropic' ? 'overloaded_error' : 'service_unavailable',
    'no free model could serve this request',
  );
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(req, res, 'openai', body);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(req, res, 'anthropic', body);
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

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
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

function sha8(text: string): string {
  return createHash('sha256').update(text).digest('hex').slice(0, 8);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models (catalog sha8 ${sha8(JSON.stringify(catalog))})`);
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
