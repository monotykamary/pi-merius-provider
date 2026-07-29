/**
 * Merius Provider Extension
 *
 * Registers Merius (merius.ai) as a custom provider using the openai-completions API.
 * Base URL: https://api.merius.ai/v1
 *
 * Merius hosts open-weight models on its own GPUs behind a single
 * OpenAI-compatible Chat Completions endpoint. The /v1/models endpoint returns
 * rich metadata per model — name, per-token pricing (prompt, completion, and an
 * optional cached-input rate), context_length, max_output_length,
 * input_modalities, and supported_features ("tools", "reasoning"). This extension
 * derives models.json directly from that metadata, so pricing, context windows,
 * vision support, and the reasoning flag self-heal on every model sync.
 *
 * Key API characteristics:
 *   - OpenAI Chat Completions compatible (/v1/chat/completions)
 *   - /v1/models is PUBLIC — the model catalog is fetched without an API key,
 *     so pi lists Merius models (and the SWR revalidation runs) even before a
 *     key is configured. A key is only required to actually call a model.
 *   - Pricing is per-token in the API; converted to per-million tokens for pi.
 *   - supported_features drives reasoning + tool support automatically.
 *
 * Reasoning compatibility:
 *   The API reports which models support reasoning, but not the wire format
 *   each model expects. Reasoning models are given the standard OpenAI-
 *   compatible defaults (thinkingFormat: "openai", supportsReasoningEffort:
 *   true, supportsDeveloperRole: false). If a model on Merius needs a different
 *   format (e.g. DeepSeek's native thinking field), override it in patch.json —
 *   see AGENTS.md.
 *
 * Model resolution strategy: Stale-While-Revalidate
 *   1. Serve stale immediately: disk cache -> embedded models.json (zero-latency)
 *   2. Revalidate in background: live API /models -> merge with embedded -> cache -> hot-swap
 *   3. patch.json + custom-models.json applied on top of whichever source won
 *
 * Merge order: [live|cache|embedded] -> apply patch.json -> merge custom-models.json
 *
 * Usage:
 *   # Option 1: Store in auth.json (recommended)
 *   # Add to ~/.pi/agent/auth.json:
 *   #   "merius": { "type": "api_key", "key": "your-api-key" }
 *
 *   # Option 2: Set as environment variable
 *   export MERIUS_API_KEY=your-api-key
 *
 *   # Run pi with the extension
 *   pi -e /path/to/pi-merius-provider
 *
 * Then use /model to select from available models like DeepSeek V4 Flash,
 * GLM-5.2, MiniMax M3, Kimi K3, and more.
 *
 * @see https://docs.merius.ai
 */

import { getAgentDir, type ExtensionAPI, type ModelRegistry } from "@earendil-works/pi-coding-agent";
import modelsData from "./models.json" with { type: "json" };
import customModelsData from "./custom-models.json" with { type: "json" };
import patchData from "./patch.json" with { type: "json" };
import deprecatedData from "./deprecated-models.json" with { type: "json" };
import fs from "fs";
import path from "path";

// ─── Types ────────────────────────────────────────────────────────────────────

interface JsonModel {
  id: string;
  name: string;
  reasoning: boolean;
  input: ("text" | "image")[];
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
  };
  contextWindow: number;
  maxTokens: number;
  thinkingLevelMap?: Record<string, string | null>;
  compat?: {
    supportsDeveloperRole?: boolean;
    supportsStore?: boolean;
    maxTokensField?: "max_completion_tokens" | "max_tokens";
    thinkingFormat?: "openai" | "openrouter" | "together" | "deepseek" | "zai" | "qwen" | "qwen-chat-template" | "string-thinking";
    supportsReasoningEffort?: boolean;
    requiresReasoningContentOnAssistantMessages?: boolean;
  };
}

interface PatchEntry {
  name?: string;
  reasoning?: boolean;
  input?: ("text" | "image")[];
  cost?: {
    input?: number;
    output?: number;
    cacheRead?: number;
    cacheWrite?: number;
  };
  contextWindow?: number;
  maxTokens?: number;
  thinkingLevelMap?: Record<string, string | null>;
  compat?: Record<string, unknown>;
}

type PatchData = Record<string, PatchEntry>;

// ─── Patch Application ────────────────────────────────────────────────────────

function applyPatch(model: JsonModel, patch: PatchEntry): JsonModel {
  const result = { ...model };

  if (patch.name !== undefined) result.name = patch.name;
  if (patch.reasoning !== undefined) result.reasoning = patch.reasoning;
  if (patch.input !== undefined) result.input = patch.input;
  if (patch.contextWindow !== undefined) result.contextWindow = patch.contextWindow;
  if (patch.maxTokens !== undefined) result.maxTokens = patch.maxTokens;
  if (patch.thinkingLevelMap !== undefined) result.thinkingLevelMap = { ...patch.thinkingLevelMap };

  if (patch.cost) {
    result.cost = {
      input: patch.cost.input ?? result.cost.input,
      output: patch.cost.output ?? result.cost.output,
      cacheRead: patch.cost.cacheRead ?? result.cost.cacheRead,
      cacheWrite: patch.cost.cacheWrite ?? result.cost.cacheWrite,
    };
  }
  if (patch.compat) {
    result.compat = { ...(result.compat || {}), ...patch.compat };
  }

  if (!result.reasoning && result.compat?.thinkingFormat) {
    delete result.compat.thinkingFormat;
  }
  if (result.compat && Object.keys(result.compat).length === 0) {
    delete result.compat;
  }

  return result;
}

/** Full pipeline: base models -> patch -> custom -> result */
function buildModels(base: JsonModel[], custom: JsonModel[], patch: PatchData): JsonModel[] {
  const modelMap = new Map<string, JsonModel>();

  for (const model of base) {
    modelMap.set(model.id, model);
  }

  for (const [id, patchEntry] of Object.entries(patch)) {
    const existing = modelMap.get(id);
    if (existing) {
      modelMap.set(id, applyPatch(existing, patchEntry));
    }
  }

  for (const model of custom) {
    const existing = modelMap.get(model.id);
    const patchEntry = patch[model.id];
    if (existing && patchEntry) {
      modelMap.set(model.id, applyPatch(model, patchEntry));
    } else if (existing) {
      modelMap.set(model.id, model);
    } else if (patchEntry) {
      modelMap.set(model.id, applyPatch(model, patchEntry));
    } else {
      modelMap.set(model.id, model);
    }
  }

  return Array.from(modelMap.values());
}

// ─── Merius /v1/models transform ──────────────────────────────────────────────
//
// Merius returns rich per-model metadata, so we derive as much as possible
// directly from the API instead of curating it by hand:
//   - name           <- apiModel.name (trailing parentheticals stripped)
//   - reasoning      <- supported_features includes "reasoning"
//   - input/vision   <- input_modalities includes "image"
//   - cost           <- pricing[0].prompt / completion (per-token -> per-million)
//   - cacheRead      <- pricing[0].input_cache_read (per-token -> per-million), if present
//   - contextWindow  <- context_length
//   - maxTokens      <- max_output_length
//
// Reasoning models get the standard OpenAI-compatible defaults (thinkingFormat
// "openai", supportsReasoningEffort true, supportsDeveloperRole false). The API
// reports *that* a model supports reasoning but not *how* it expects it on the
// wire; patch.json overrides the format for any model that needs something else
// (e.g. a DeepSeek-native thinking field).

const PER_TOKEN_TO_PER_MILLION = 1_000_000;
// Clean float noise (e.g. 0.0000003 * 1e6 -> 0.30000000000000004) without losing
// micro-dollar precision on small cache rates.
const COST_ROUND_FACTOR = 1_000_000;

/** Convert a per-token price (string or number from the API) to per-million. */
function toPerMillion(perToken: string | number | undefined | null): number {
  if (perToken === undefined || perToken === null) return 0;
  const n = typeof perToken === "number" ? perToken : parseFloat(String(perToken));
  if (!Number.isFinite(n) || n <= 0) return 0;
  const perMillion = n * PER_TOKEN_TO_PER_MILLION;
  return Math.round(perMillion * COST_ROUND_FACTOR) / COST_ROUND_FACTOR;
}

function cleanDisplayName(name: unknown, id: string): string {
  if (typeof name === "string" && name.trim().length > 0) {
    // Strip a trailing parenthetical (e.g. "GLM-5.2 (default: FP4 — fast + cheap)")
    const stripped = name.replace(/\s*\([^)]*\)\s*$/, "").trim();
    if (stripped.length > 0) return stripped;
  }
  const parts = id.split("/");
  const rawName = parts.length > 1 ? parts.slice(1).join("/") : id;
  return rawName.replace(/-/g, " ").replace(/\b\w/g, (c: string) => c.toUpperCase());
}

function transformApiModel(apiModel: any): JsonModel | null {
  if (!apiModel || typeof apiModel.id !== "string" || apiModel.id.length === 0) return null;

  const features: string[] = Array.isArray(apiModel.supported_features) ? apiModel.supported_features : [];
  const reasoning = features.some((f) => typeof f === "string" && f.toLowerCase() === "reasoning");

  const inputMods: string[] = Array.isArray(apiModel.input_modalities) ? apiModel.input_modalities : ["text"];
  const hasImage = inputMods.some((m) => typeof m === "string" && m.toLowerCase().includes("image"));

  // pricing is an array of per-token rate cards; take the first (standard tier).
  const pricingEntry =
    Array.isArray(apiModel.pricing) && apiModel.pricing.length > 0 && typeof apiModel.pricing[0] === "object"
      ? apiModel.pricing[0]
      : {};

  // Accept either OpenAI-style (prompt/completion) or Merius alt spellings.
  const inputCost = toPerMillion(pricingEntry.prompt ?? pricingEntry.input);
  const outputCost = toPerMillion(pricingEntry.completion ?? pricingEntry.output);
  const cacheRead = toPerMillion(pricingEntry.input_cache_read ?? pricingEntry.cache_read);

  const contextLength = Number(apiModel.context_length);
  const maxOutputLength = Number(apiModel.max_output_length);
  const contextWindow = Number.isFinite(contextLength) && contextLength > 0 ? contextLength : 131072;
  const maxTokens =
    Number.isFinite(maxOutputLength) && maxOutputLength > 0 ? maxOutputLength : contextWindow;

  const model: JsonModel = {
    id: apiModel.id,
    name: cleanDisplayName(apiModel.name, apiModel.id),
    reasoning,
    input: hasImage ? ["text", "image"] : ["text"],
    cost: {
      input: inputCost,
      output: outputCost,
      cacheRead,
      cacheWrite: 0,
    },
    contextWindow,
    maxTokens,
  };

  if (reasoning) {
    model.compat = {
      thinkingFormat: "openai",
      supportsReasoningEffort: true,
      supportsDeveloperRole: false,
    };
    model.thinkingLevelMap = {
      minimal: null,
      low: null,
      medium: null,
      high: "high",
      max: "max",
    };
  }

  return model;
}

// ─── Stale-While-Revalidate Model Sync ────────────────────────────────────────

const PROVIDER_ID = "merius";
const BASE_URL = "https://api.merius.ai/v1";
const MODELS_URL = `${BASE_URL}/models`;
const CACHE_DIR = path.join(getAgentDir(), "cache");
const CACHE_PATH = path.join(CACHE_DIR, `${PROVIDER_ID}-models.json`);
const LIVE_FETCH_TIMEOUT_MS = 8000;

async function fetchLiveModels(apiKey: string | undefined, signal?: AbortSignal): Promise<JsonModel[] | null> {
  try {
    const response = await fetch(MODELS_URL, {
      headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
      signal: signal ? AbortSignal.any([AbortSignal.timeout(LIVE_FETCH_TIMEOUT_MS), signal]) : AbortSignal.timeout(LIVE_FETCH_TIMEOUT_MS),
    });
    if (!response.ok) return null;
    const data = await response.json();
    const apiModels = Array.isArray(data) ? data : (data.data || []);
    if (!Array.isArray(apiModels) || apiModels.length === 0) return null;
    return apiModels.map(transformApiModel).filter((m): m is JsonModel => m !== null);
  } catch {
    return null;
  }
}

function loadCachedModels(): JsonModel[] | null {
  try {
    const data = JSON.parse(fs.readFileSync(CACHE_PATH, "utf8"));
    return Array.isArray(data) ? data : null;
  } catch {
    return null;
  }
}

function cacheModels(models: JsonModel[]): void {
  try {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
    fs.writeFileSync(CACHE_PATH, JSON.stringify(models, null, 2) + "\n");
  } catch {
    // Cache write failure is non-fatal
  }
}

function mergeWithEmbedded(liveModels: JsonModel[], embeddedModels: JsonModel[]): JsonModel[] {
  const embeddedMap = new Map(embeddedModels.map(m => [m.id, m]));
  const seen = new Set<string>();
  const result: JsonModel[] = [];
  for (const liveModel of liveModels) {
    const embedded = embeddedMap.get(liveModel.id);
    seen.add(liveModel.id);
    if (embedded) {
      // Self-heal: live API pricing is authoritative field-by-field. Prefer the
      // live cost when the API reports it (non-zero); fall back to embedded when
      // the API is silent (0) so curated cacheRead/cacheWrite isn't clobbered and
      // providers whose /models endpoint exposes no pricing keep their curated
      // cost. Curation (reasoning/input/compat/name) still wins via ...embedded.
      result.push({
        ...liveModel,
        ...embedded,
        cost: {
          input: liveModel.cost.input || embedded.cost.input,
          output: liveModel.cost.output || embedded.cost.output,
          cacheRead: liveModel.cost.cacheRead || embedded.cost.cacheRead,
          cacheWrite: liveModel.cost.cacheWrite || embedded.cost.cacheWrite,
        },
        contextWindow: liveModel.contextWindow || embedded.contextWindow,
      });
    } else {
      result.push(liveModel);
    }
  }
  for (const em of embeddedModels) {
    if (!seen.has(em.id)) {
      result.push(em);
    }
  }
  return result;
}

// Grace period for delisted models. When the provider API stops listing a
// model, update-models.js moves its last-known definition into
// deprecated-models.json (stamped with deprecatedAt) instead of dropping it.
// For 14 days the model keeps working here so in-flight sessions and saved
// model settings do not break; afterwards it is evicted permanently.
const DEPRECATED_MODEL_TTL_MS = 14 * 24 * 60 * 60 * 1000;

// Grace-period deprecated models with deprecation metadata stripped.
function activeDeprecatedModels(): JsonModel[] {
  const now = Date.now();
  const result: JsonModel[] = [];
  for (const entry of Object.values(deprecatedData as Record<string, JsonModel & { deprecatedAt?: string }>)) {
    if (!entry?.id) continue;
    const removedAt = Date.parse(entry.deprecatedAt ?? "");
    if (Number.isNaN(removedAt) || now - removedAt > DEPRECATED_MODEL_TTL_MS) continue;
    const model = { ...entry } as JsonModel & { deprecatedAt?: string };
    delete model.deprecatedAt;
    result.push(model);
  }
  return result;
}

// Append grace-period deprecated models the list does not already have (live data wins).
function withDeprecated(models: JsonModel[]): JsonModel[] {
  const seen = new Set(models.map((m) => m.id));
  const extras = activeDeprecatedModels().filter((m) => !seen.has(m.id));
  return extras.length > 0 ? [...models, ...extras] : models;
}

function loadStaleModels(embeddedModels: JsonModel[]): JsonModel[] {
  const cached = loadCachedModels();
  if (!cached || cached.length === 0) return embeddedModels;

  const cachedMap = new Map(cached.map(m => [m.id, m]));
  for (const em of embeddedModels) {
    if (!cachedMap.has(em.id)) {
      cached.push(em);
    }
  }
  return cached;
}

async function revalidateModels(apiKey: string | undefined, embeddedModels: JsonModel[], signal?: AbortSignal): Promise<JsonModel[] | null> {
  // /v1/models is public on Merius — revalidate even without an API key so the
  // catalogue stays fresh before a key is configured. A key is only needed to
  // actually call a model. The Authorization header is still sent when a key is
  // available, in case the endpoint becomes auth-gated later.
  const liveModels = await fetchLiveModels(apiKey, signal);
  if (!liveModels || liveModels.length === 0) return null;
  const merged = mergeWithEmbedded(liveModels, embeddedModels);
  cacheModels(merged);
  return merged;
}

// ─── API Key Resolution (via ModelRegistry) ────────────────────────────────────

let revalidateAbort: AbortController | null = null;

// Key is only resolved so the provider registration can reference it; the SWR
// fetch itself needs no key (see revalidateModels).
async function resolveApiKey(modelRegistry: ModelRegistry): Promise<string | undefined> {
  return (await modelRegistry.getApiKeyForProvider("merius")) ?? undefined;
}

// ─── Extension Entry Point ────────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
  const embeddedModels = modelsData as JsonModel[];
  const customModels = customModelsData as JsonModel[];
  const patches = patchData as PatchData;

  const staleBase = loadStaleModels(embeddedModels);
  const staleModels = buildModels(staleBase, customModels, patches);

  pi.registerProvider("merius", {
    baseUrl: BASE_URL,
    apiKey: "$MERIUS_API_KEY",
    api: "openai-completions",
    models: withDeprecated(staleModels),
  });

  pi.on("session_start", async (_event, ctx) => {
    revalidateAbort?.abort();
    revalidateAbort = new AbortController();
    const signal = revalidateAbort.signal;
    // /v1/models is public, so revalidation proceeds with or without a key; the
    // key is still threaded through so the Bearer header is sent when available.
    resolveApiKey(ctx.modelRegistry).then((apiKey) => {
      revalidateModels(apiKey, embeddedModels, signal).then((freshBase) => {
        if (freshBase && !signal.aborted) {
          pi.registerProvider("merius", {
            baseUrl: BASE_URL,
            apiKey: "$MERIUS_API_KEY",
            api: "openai-completions",
            models: withDeprecated(buildModels(freshBase, customModels, patches)),
          });
        }
      });
    });
  });

  pi.on("session_shutdown", () => {
    revalidateAbort?.abort();
  });
}
