#!/usr/bin/env node
/**
 * Update Tarmis models from API
 *
 * Fetches models from https://api.tarmis.ai/v1/models (public — no API key
 * required) and updates:
 * - models.json: model definitions built from the API's rich metadata
 * - README.md: Model table with patch.json overrides applied
 *
 * Tarmis /v1/models returns per-model metadata (name, per-token pricing,
 * context_length, max_output_length, input_modalities, supported_features, etc.).
 * transformModel derives reasoning, vision, pricing, context window, and max
 * output directly from that metadata. patch.json overrides anything the API
 * can't tell us (e.g. a model's reasoning wire format).
 *
 * Merge order for README: models.json -> apply patch.json -> merge custom-models.json
 *
 * API key: optional. The stored `tarmis` credential in ~/.pi/agent/auth.json (or
 * TARMIS_API_KEY) is sent when it resolves; the public endpoint returned the full
 * catalog unauthenticated at last check, so running without one is acceptable.
 */

import fs from 'fs';
import os from 'os';
import { execSync } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// pi's agent directory: PI_CODING_AGENT_DIR (with ~ expansion) or ~/.pi/agent.
function piAgentDir() {
  const envDir = process.env.PI_CODING_AGENT_DIR;
  if (envDir) {
    return envDir.startsWith('~/') || envDir === '~'
      ? path.join(os.homedir(), envDir.slice(1))
      : envDir;
  }
  return path.join(os.homedir(), '.pi', 'agent');
}

const AUTH_JSON_PATH = path.join(piAgentDir(), 'auth.json');

/**
 * Resolve a configured value using pi's semantics (resolve-config-value.ts in
 * pi-mono): "!command" runs via the shell (10s timeout) and uses trimmed
 * stdout; "$VAR" / "${VAR}" interpolate environment variables ("$$" escapes a
 * literal "$", "$!" a literal "!"); anything else is a literal. Returns
 * undefined when a referenced env var is unset or a command fails.
 */
function resolveConfigValue(config, env) {
  if (typeof config !== 'string' || config.length === 0) return undefined;
  if (config.startsWith('!')) {
    try {
      const out = execSync(config.slice(1), {
        encoding: 'utf8',
        timeout: 10000,
        stdio: ['ignore', 'pipe', 'ignore'],
      });
      return out.trim() || undefined;
    } catch {
      return undefined;
    }
  }
  const ENV_NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
  let resolved = '';
  let index = 0;
  while (index < config.length) {
    const dollar = config.indexOf('$', index);
    if (dollar < 0) {
      resolved += config.slice(index);
      break;
    }
    resolved += config.slice(index, dollar);
    const next = config[dollar + 1];
    let name;
    if (next === '$' || next === '!') {
      resolved += next;
      index = dollar + 2;
      continue;
    } else if (next === '{') {
      const end = config.indexOf('}', dollar + 2);
      if (end < 0) {
        resolved += '$';
        index = dollar + 1;
        continue;
      }
      const inner = config.slice(dollar + 2, end);
      if (!ENV_NAME_RE.test(inner)) {
        resolved += config.slice(dollar, end + 1);
        index = end + 1;
        continue;
      }
      name = inner;
      index = end + 1;
    } else {
      const match = config.slice(dollar + 1).match(/^[A-Za-z_][A-Za-z0-9_]*/);
      if (!match) {
        resolved += '$';
        index = dollar + 1;
        continue;
      }
      name = match[0];
      index = dollar + 1 + name.length;
    }
    const value = (env && env[name]) || process.env[name] || undefined;
    if (value === undefined) return undefined;
    resolved += value;
  }
  return resolved;
}

/**
 * The API key, resolved the way pi itself resolves it for this provider: the
 * stored `tarmis` credential in ~/.pi/agent/auth.json wins, then
 * the TARMIS_API_KEY environment variable.
 */
function resolveApiKey() {
  try {
    const auth = JSON.parse(fs.readFileSync(AUTH_JSON_PATH, 'utf8'));
    const credential = auth?.tarmis;
    if (credential && credential.type === 'api_key' && typeof credential.key === 'string') {
      const key = resolveConfigValue(credential.key, credential.env);
      if (key) return key;
    }
  } catch {
    // Missing or unparseable auth.json: fall through to the env var.
  }
  return process.env.TARMIS_API_KEY || undefined;
}

const MODELS_API_URL = 'https://api.tarmis.ai/v1/models';
const MODELS_JSON_PATH = path.join(__dirname, '..', 'models.json');
const PATCH_JSON_PATH = path.join(__dirname, '..', 'patch.json');
const CUSTOM_MODELS_JSON_PATH = path.join(__dirname, '..', 'custom-models.json');
const README_PATH = path.join(__dirname, '..', 'README.md');

const PER_TOKEN_TO_PER_MILLION = 1_000_000;
const COST_ROUND_FACTOR = 1_000_000;

// ─── Patch application ────────────────────────────────────────────────────────

function applyPatch(model, patch) {
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

function buildModels(baseModels, customModels, patchData) {
  const modelMap = new Map();

  for (const model of baseModels) {
    modelMap.set(model.id, model);
  }

  for (const [id, patchEntry] of Object.entries(patchData)) {
    const existing = modelMap.get(id);
    if (existing) {
      modelMap.set(id, applyPatch(existing, patchEntry));
    }
  }

  for (const model of customModels) {
    const existing = modelMap.get(model.id);
    const patchEntry = patchData[model.id];
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

// ─── Tarmis /v1/models transform ─────────────────────────────────────────────

function toPerMillion(perToken) {
  if (perToken === undefined || perToken === null) return 0;
  const n = typeof perToken === 'number' ? perToken : parseFloat(String(perToken));
  if (!Number.isFinite(n) || n <= 0) return 0;
  const perMillion = n * PER_TOKEN_TO_PER_MILLION;
  return Math.round(perMillion * COST_ROUND_FACTOR) / COST_ROUND_FACTOR;
}

function cleanDisplayName(name, id) {
  if (typeof name === 'string' && name.trim().length > 0) {
    const stripped = name.replace(/\s*\([^)]*\)\s*$/, '').trim();
    if (stripped.length > 0) return stripped;
  }
  const parts = id.split('/');
  const rawName = parts.length > 1 ? parts.slice(1).join('/') : id;
  return rawName.replace(/-/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

function transformModel(apiModel) {
  if (!apiModel || typeof apiModel.id !== 'string' || apiModel.id.length === 0) return null;

  const features = Array.isArray(apiModel.supported_features) ? apiModel.supported_features : [];
  const reasoning = features.some((f) => typeof f === 'string' && f.toLowerCase() === 'reasoning');

  const inputMods = Array.isArray(apiModel.input_modalities) ? apiModel.input_modalities : ['text'];
  const hasImage = inputMods.some((m) => typeof m === 'string' && m.toLowerCase().includes('image'));

  const pricingEntry =
    Array.isArray(apiModel.pricing) && apiModel.pricing.length > 0 && typeof apiModel.pricing[0] === 'object'
      ? apiModel.pricing[0]
      : {};

  const inputCost = toPerMillion(pricingEntry.prompt ?? pricingEntry.input);
  const outputCost = toPerMillion(pricingEntry.completion ?? pricingEntry.output);
  const cacheRead = toPerMillion(pricingEntry.input_cache_read ?? pricingEntry.cache_read);

  const isFree = apiModel.is_free === true || (inputCost === 0 && outputCost === 0);

  const contextLength = Number(apiModel.context_length);
  const maxOutputLength = Number(apiModel.max_output_length);
  const contextWindow = Number.isFinite(contextLength) && contextLength > 0 ? contextLength : 131072;
  const maxTokens =
    Number.isFinite(maxOutputLength) && maxOutputLength > 0 ? maxOutputLength : contextWindow;

  const model = {
    id: apiModel.id,
    name: cleanDisplayName(apiModel.name, apiModel.id),
    reasoning,
    input: hasImage ? ['text', 'image'] : ['text'],
    cost: { input: inputCost, output: outputCost, cacheRead, cacheWrite: 0 },
    contextWindow,
    maxTokens,
    _meta: { isFree },
  };

  if (reasoning) {
    model.compat = {
      thinkingFormat: 'openai',
      supportsReasoningEffort: true,
      supportsDeveloperRole: false,
    };
    model.thinkingLevelMap = { minimal: null, low: null, medium: null, high: 'high', max: 'max' };
  }

  return model;
}

// ─── File I/O ─────────────────────────────────────────────────────────────────

function loadJson(filePath, fallback) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return fallback;
  }
}

// ─── README generation ────────────────────────────────────────────────────────

function formatCost(cost, isFree) {
  if (isFree) return '**Free**';
  if (cost === 0) return '-';
  return `$${cost.toFixed(2)}`;
}

function formatContextWindow(n) {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (n >= 1000) return `${Math.round(n / 1000)}K`;
  return n.toString();
}

function generateReadmeTable(models) {
  const lines = [
    '| Model | Context | Vision | Reasoning | Input $/M | Output $/M |',
    '|-------|---------|--------|-----------|-----------|------------|',
  ];

  for (const model of models) {
    const name = model.name;
    const context = formatContextWindow(model.contextWindow);
    const vision = model.input.includes('image') ? '✅' : '❌';
    const reasoning = model.reasoning ? '✅' : '❌';
    const isFree = model._meta?.isFree ?? (model.cost.input === 0 && model.cost.output === 0);
    const inputCost = formatCost(model.cost.input, isFree);
    const outputCost = formatCost(model.cost.output, isFree);

    lines.push(`| ${name} | ${context} | ${vision} | ${reasoning} | ${inputCost} | ${outputCost} |`);
  }

  return lines.join('\n');
}

function updateReadme(models) {
  let readme = fs.readFileSync(README_PATH, 'utf8');
  const newTable = generateReadmeTable(models);

  const tableRegex = /(## Available Models\n\n)\| Model \| Context \| Vision \| Reasoning \| Input \$\/M \| Output \$\/M \|\n\|[-| ]+\|(\n\|[^\n]+\|)*\n*/;

  if (tableRegex.test(readme)) {
    readme = readme.replace(tableRegex, (match, header) => `${header}${newTable}\n\n`);
    fs.writeFileSync(README_PATH, readme);
    console.log('✓ Updated README.md');
  } else {
    console.warn('⚠ Could not find model table in "## Available Models" section');
  }
}

function cleanModelForJson(model) {
  const { _meta, ...cleanModel } = model;
  return cleanModel;
}

// ─── Main ──────────────────────────────────────────────────────────────────────

// Grace period for delisted models: update-models.js moves models the API no
// longer lists into deprecated-models.json (stamped with deprecatedAt) instead
// of dropping them; the runtime appends them back so sessions and saved model
// settings keep working, and after 14 days they are evicted permanently.
const DEPRECATED_MODEL_TTL_MS = 14 * 24 * 60 * 60 * 1000;

/**
 * Reconcile deprecated-models.json against the freshly fetched model list.
 * - in old models.json but not the API: moved into the deprecated file
 *   (deprecatedAt = now; preserved on repeat runs so the grace clock is not reset)
 * - back in the API: resurrected (dropped from the deprecated file)
 * - deprecatedAt older than 14 days: evicted permanently
 * Must run BEFORE the new models.json is written; it reads the old file itself.
 */
function updateDeprecatedModels(modelsJsonPath, newModels) {
  const deprecatedPath = path.join(path.dirname(modelsJsonPath), 'deprecated-models.json');

  let oldModels = [];
  try {
    const parsed = JSON.parse(fs.readFileSync(modelsJsonPath, 'utf8'));
    if (Array.isArray(parsed)) oldModels = parsed;
  } catch { /* first run: no previous models.json */ }

  let deprecated = {};
  try {
    const parsed = JSON.parse(fs.readFileSync(deprecatedPath, 'utf8'));
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) deprecated = parsed;
  } catch { /* no graveyard yet */ }

  const currentIds = new Set(newModels.map((m) => m.id));
  const now = new Date().toISOString();
  const added = [];
  const resurrected = [];
  const evicted = [];

  for (const old of oldModels) {
    if (old && old.id && !currentIds.has(old.id) && !deprecated[old.id]) {
      deprecated[old.id] = { ...old, deprecatedAt: now };
      added.push(old.id);
    }
  }

  for (const [id, entry] of Object.entries(deprecated)) {
    if (currentIds.has(id)) {
      delete deprecated[id];
      resurrected.push(id);
      continue;
    }
    const removedAt = Date.parse(entry && entry.deprecatedAt ? entry.deprecatedAt : '');
    if (Number.isNaN(removedAt) || Date.now() - removedAt > DEPRECATED_MODEL_TTL_MS) {
      delete deprecated[id];
      evicted.push(id);
    }
  }

  if (added.length > 0 || resurrected.length > 0 || evicted.length > 0) {
    fs.writeFileSync(deprecatedPath, JSON.stringify(deprecated, null, 2) + '\n');
    console.log('Updated deprecated-models.json ' + JSON.stringify({ added, resurrected, evicted }));
  }
}

async function main() {
  console.log(`Fetching models from ${MODELS_API_URL}...`);

  try {
    const apiKey = resolveApiKey();
    // Tarmis /v1/models is public and complete (verified against the curated
    // catalog); send the credential when one resolves, proceed without it.
    const headers = apiKey ? { Authorization: `Bearer ${apiKey}` } : {};

    const response = await fetch(MODELS_API_URL, { headers });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const apiResponse = await response.json();
    const apiModels = apiResponse.data || apiResponse;

    if (!Array.isArray(apiModels)) {
      throw new Error('API response does not contain an array of models');
    }

    console.log(`✓ Fetched ${apiModels.length} models from API`);

    // Transform models from API metadata (filter out anything unparseable)
    let apiTransformed = apiModels
      .map(transformModel)
      .filter((m) => m !== null);
    console.log(`✓ Transformed ${apiTransformed.length} chat models`);

    // Sort by id for stable diffs
    apiTransformed.sort((a, b) => a.id.localeCompare(b.id));

    // Load existing models.json — used to flag what's new, and as the prior
    // snapshot for deprecated-models.json reconciliation.
    let existingModels = [];
    try {
      existingModels = JSON.parse(fs.readFileSync(MODELS_JSON_PATH, 'utf8'));
    } catch {
      // File might not exist or be invalid
    }
    const existingModelsMap = {};
    for (const m of existingModels) {
      existingModelsMap[m.id] = m;
    }

    // Update models.json — API-derived model list
    const cleanModels = apiTransformed.map(cleanModelForJson);
    // Move delisted models to deprecated-models.json BEFORE models.json is overwritten
    updateDeprecatedModels(MODELS_JSON_PATH, cleanModels);
    fs.writeFileSync(MODELS_JSON_PATH, JSON.stringify(cleanModels, null, 2) + '\n');
    console.log('✓ Updated models.json (API-derived model list)');

    // Log new models not yet in patch.json
    const patch = loadJson(PATCH_JSON_PATH, {});
    for (const m of apiTransformed) {
      if (!patch[m.id] && !existingModelsMap[m.id]) {
        const feature = m.reasoning ? ' (reasoning)' : '';
        console.log(`  🆕 New model: ${m.id} (${m.name})${feature} — add to patch.json for reasoning compat overrides if needed`);
      }
    }

    // Build full model list for README: base -> patch -> custom
    const customModels = loadJson(CUSTOM_MODELS_JSON_PATH, []);
    const readmeModels = buildModels(
      apiTransformed,
      Array.isArray(customModels) ? customModels : [],
      patch
    );
    readmeModels.sort((a, b) => a.name.localeCompare(b.name));
    console.log('✓ Built model list (base -> patch -> custom) for README');

    // Update README.md with patched data
    updateReadme(readmeModels);

    // Summary
    console.log('\n--- Summary ---');
    console.log(`Total models: ${readmeModels.length}`);
    console.log(`Reasoning models: ${readmeModels.filter((m) => m.reasoning).length}`);
    console.log(`Vision models: ${readmeModels.filter((m) => m.input.includes('image')).length}`);

    const newIds = new Set(apiTransformed.map((m) => m.id));
    const oldIds = new Set(existingModels.map((m) => m.id));

    const added = [...newIds].filter((id) => !oldIds.has(id));
    const removed = [...oldIds].filter((id) => !newIds.has(id));

    if (added.length > 0) {
      console.log(`\nNew models: ${added.join(', ')}`);
    }
    if (removed.length > 0) {
      console.log(`\nRemoved models: ${removed.join(', ')}`);
    }
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

main();
