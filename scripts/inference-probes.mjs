#!/usr/bin/env node
/*
 * inference-probes.mjs — lean provider-quality probes for Merius, complementing synbad.
 *
 * Covers what synbad doesn't:
 *   P1  reasoning field shape (reasoning vs reasoning_content) + preserved-thinking round-trip
 *   P2  preserved thinking on tool-call assistant messages (the Kimi-style 400 bug)
 *   P3  strict json_schema grammar constraint (escapes into reasoning, 400s)
 *   P4  sampling/logit acceptance: stop strings, logprobs, logit_bias
 *   P5  vision: single-image QA + 24-image burst on Kimi K3; graceful rejection elsewhere
 *
 * Usage: MERIUS_API_KEY=... node scripts/inference-probes.mjs [--only <modelId>]
 */

import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";

const BASE = process.env.MERIUS_BASE_URL ?? "https://api.merius.ai/v1";
const KEY = process.env.MERIUS_API_KEY;
if (!KEY) { console.error("MERIUS_API_KEY env var required"); process.exit(2); }

const ALL_MODELS = [
  "z-ai/glm-5.2",
  "minimax/minimax-m3",
  "deepseek-ai/DeepSeek-V4-Flash",
  "moonshotai/Kimi-K3",
];

const only = (() => {
  const i = process.argv.indexOf("--only");
  return i >= 0 ? process.argv[i + 1] : null;
})();
const MODELS = only ? ALL_MODELS.filter((m) => m.toLowerCase().includes(only.toLowerCase())) : ALL_MODELS;

// minimal solid-color PNG writer — no external deps
const crcTable = new Int32Array(256);
for (let n = 0; n < 256; n++) {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  crcTable[n] = c;
}
function crc32(buf) {
  let c = -1;
  for (let i = 0; i < buf.length; i++) c = crcTable[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}
function pngChunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
}
function solidPng(size, [r, g, b]) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8;  // bit depth
  ihdr[9] = 2;  // truecolor RGB
  const row = Buffer.alloc(1 + size * 3);
  for (let x = 0; x < size; x++) {
    row[1 + x * 3] = r;
    row[2 + x * 3] = g;
    row[3 + x * 3] = b;
  }
  const raw = Buffer.concat(Array.from({ length: size }, () => row));
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", zlib.deflateSync(raw)),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
}
const COLORS = {
  red: [220, 30, 30], green: [30, 180, 60], blue: [30, 80, 220], yellow: [230, 210, 40],
  magenta: [200, 30, 200], cyan: [30, 200, 200], orange: [240, 140, 20], purple: [120, 40, 160],
};

const results = [];
function record(model, probe, pass, note, extra) {
  results.push({ model, probe, pass, note, extra });
  console.log(`${pass === true ? "PASS" : pass === "skip" ? "SKIP" : "FAIL"}  ${model}  ${probe}  ${note}`);
}

async function chat(model, body) {
  let res;
  try {
    res = await fetch(`${BASE}/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${KEY}`, "content-type": "application/json" },
      body: JSON.stringify({ model, ...body }),
      signal: AbortSignal.timeout(240_000),
    });
  } catch (e) {
    return { status: 0, json: null, err: String(e).slice(0, 200) };
  }
  const text = await res.text();
  let json = null;
  try { json = JSON.parse(text); } catch {}
  return { status: res.status, json, text: json ? undefined : text.slice(0, 400), raw: text.slice(0, 400) };
}

// retry once without reasoning_effort if the backend rejects it
async function chatTolerant(model, body) {
  let r = await chat(model, body);
  if (r.status >= 400 && body.reasoning_effort && /reasoning_effort/i.test(r.text ?? r.raw ?? "")) {
    const { reasoning_effort, ...rest } = body;
    r = await chat(model, rest);
  }
  return r;
}

function msgOf(r) { return r.json?.choices?.[0]?.message ?? null; }
function reasoningOf(m) { return m?.reasoning_content ?? m?.reasoning ?? null; }
function reasoningKeyOf(m) { return m?.reasoning_content != null ? "reasoning_content" : m?.reasoning != null ? "reasoning" : null; }
function briefErr(r) {
  const t = r.text ?? r.raw ?? r.err ?? JSON.stringify(r.json)?.slice(0, 200) ?? "";
  return `HTTP ${r.status} ${String(t).slice(0, 180)}`;
}

const WEATHER_TOOL = {
  type: "function",
  function: {
    name: "get_weather",
    description: "Get current weather for a location",
    parameters: {
      type: "object",
      properties: { location: { type: "string", description: "City name" } },
      required: ["location"],
    },
  },
};

// P1: reasoning shape + preserved-thinking round-trip
async function pReasoning(model) {
  const u1 = { role: "user", content: "What is 23 * 47? Reply with only the number." };
  const t1 = await chatTolerant(model, { messages: [u1], max_tokens: 1024, reasoning_effort: "low" });
  const m1 = msgOf(t1);
  if (!m1) return record(model, "P1.reasoning-roundtrip", false, `turn1 failed: ${briefErr(t1)}`);
  const key = reasoningKeyOf(m1);
  record(model, "P1.reasoning-shape", "skip", key ? `thinking returned as '${key}'` : "no reasoning field returned");
  if (!/1[ ,]?081/.test(m1.content ?? "")) {
    record(model, "P1.answer", false, `expected 1081, got: ${(m1.content ?? "").slice(0, 60)}`);
  }

  const u2 = { role: "user", content: "Multiply your previous answer by 1000. Reply with only the number." };
  const asstWith = { role: "assistant", content: m1.content ?? "", ...(key ? { [key]: reasoningOf(m1) } : {}) };
  const asstWithout = { role: "assistant", content: m1.content ?? "" };
  const altKey = key === "reasoning" ? "reasoning_content" : "reasoning";

  const tWith = await chatTolerant(model, { messages: [u1, asstWith, u2], max_tokens: 1024, reasoning_effort: "low" });
  record(model, "P1.preserved-thinking", tWith.status === 200,
    tWith.status === 200 ? `accepted '${key}' echo` : `rejected preserved '${key}': ${briefErr(tWith)}`);

  const tWithout = await chatTolerant(model, { messages: [u1, asstWithout, u2], max_tokens: 1024, reasoning_effort: "low" });
  record(model, "P1.stripped-thinking", tWithout.status === 200,
    tWithout.status === 200 ? "stripped thinking accepted (lenient)" : `requires preserved thinking! ${briefErr(tWithout)}`);

  if (key && reasoningOf(m1)) {
    const asstAlt = { role: "assistant", content: m1.content ?? "", [altKey]: reasoningOf(m1) };
    const tAlt = await chatTolerant(model, { messages: [u1, asstAlt, u2], max_tokens: 1024, reasoning_effort: "low" });
    record(model, "P1.alt-key-thinking", tAlt.status === 200,
      tAlt.status === 200 ? `accepted rival '${altKey}' key too` : `rejects '${altKey}': ${briefErr(tAlt)}`);
  }
}

// P2: preserved thinking on tool-call assistant messages (Kimi-style strictness)
async function pToolsPreserved(model) {
  const u1 = { role: "user", content: "What's the weather in Paris? Use the tool." };
  const t1 = await chatTolerant(model, {
    messages: [u1], tools: [WEATHER_TOOL],
    tool_choice: { type: "function", function: { name: "get_weather" } },
    max_tokens: 1024, reasoning_effort: "low",
  });
  const m1 = msgOf(t1);
  if (!m1?.tool_calls?.length) return record(model, "P2.tools-roundtrip", false, `turn1 no tool call: ${briefErr(t1)}`);
  const key = reasoningKeyOf(m1);
  const toolMsg = { role: "tool", tool_call_id: m1.tool_calls[0].id, content: "22°C, sunny in Paris." };
  const u2 = { role: "user", content: "Thanks. In one word: is that warm?" };

  for (const [label, keep] of [["preserved", true], ["stripped", false]]) {
    const asst = {
      role: "assistant", content: m1.content ?? "", tool_calls: m1.tool_calls,
      ...(keep && key ? { [key]: reasoningOf(m1) } : {}),
    };
    const t2 = await chatTolerant(model, { messages: [u1, asst, toolMsg, u2], tools: [WEATHER_TOOL], max_tokens: 1024, reasoning_effort: "low" });
    record(model, `P2.tool-msg-thinking-${label}`, t2.status === 200,
      t2.status === 200 ? "accepted" : `${label} reasoning on tool-call msg rejected: ${briefErr(t2)}`);
  }
}

// P3: strict json_schema grammar
async function pGrammar(model) {
  const schema = {
    type: "json_schema",
    json_schema: {
      name: "calc_result", strict: true,
      schema: {
        type: "object",
        properties: {
          sum: { type: "integer" },
          parity: { type: "string", enum: ["even", "odd"] },
          unit: { type: "string", enum: ["widgets"] },
        },
        required: ["sum", "parity", "unit"],
        additionalProperties: false,
      },
    },
  };
  const r = await chatTolerant(model, {
    messages: [{ role: "user", content: "Compute 12345 + 67890 widgets." }],
    response_format: schema, max_tokens: 1024, reasoning_effort: "low",
  });
  if (r.status !== 200) {
    if (r.status === 400 || r.status === 422) {
      const r2 = await chatTolerant(model, {
        messages: [{ role: "user", content: "Compute 12345 + 67890 widgets. Answer as JSON object with fields: integer sum, parity (\"even\"|\"odd\"), unit." }],
        response_format: { type: "json_object" }, max_tokens: 1024, reasoning_effort: "low",
      });
      return record(model, "P3.grammar-jsonschema", false, `json_schema rejected (${briefErr(r)}); json_object ${r2.status === 200 ? "accepted" : briefErr(r2)}`);
    }
    return record(model, "P3.grammar-jsonschema", false, `failed: ${briefErr(r)}`);
  }
  const m = msgOf(r);
  let parsed = null;
  try { parsed = JSON.parse(m?.content ?? ""); } catch {}
  if (!parsed) return record(model, "P3.grammar-jsonschema", false, `content not valid JSON: ${(m?.content ?? "").slice(0, 80)}`);
  const keysOk = ["sum", "parity", "unit"].every((k) => k in parsed);
  const ok = typeof parsed.sum === "number" && ["even", "odd"].includes(parsed.parity) && parsed.unit === "widgets" && keysOk;
  const leak = /\$schema|additionalProperties|"required"|"properties"/.test(reasoningOf(m) ?? "");
  record(model, "P3.grammar-jsonschema", ok,
    `conformant=${ok} sum=${parsed.sum} parity=${parsed.parity}${leak ? " (WARN: schema tokens leaked into thinking)" : ""}`);
  if (parsed.sum !== 80235) record(model, "P3.grammar-accuracy", false, `sum=${parsed.sum}, expected 80235`);
}

// P4: sampling/logit acceptance
async function pSampling(model) {
  const stop = await chatTolerant(model, {
    messages: [{ role: "user", content: "Output exactly: ALPHA BETA FINAL GAMMA" }],
    stop: ["FINAL"], max_tokens: 512, reasoning_effort: "low",
  });
  const c = msgOf(stop)?.content ?? "";
  const fin = stop.json?.choices?.[0]?.finish_reason;
  record(model, "P4.stop-string", stop.status === 200 && !c.includes("FINAL") && ["stop", null, undefined].includes(fin),
    stop.status !== 200 ? briefErr(stop) : `finish_reason=${fin} content=${JSON.stringify(c.slice(0, 40))}`);

  const lp = await chatTolerant(model, {
    messages: [{ role: "user", content: "Say hi." }],
    logprobs: true, top_logprobs: 3, max_tokens: 256, reasoning_effort: "low",
  });
  const hasLp = !!lp.json?.choices?.[0]?.logprobs?.content?.length;
  record(model, "P4.logprobs", lp.status === 200,
    lp.status === 200 ? (hasLp ? "logprobs returned" : "accepted but logprobs absent (silently ignored)") : `rejected: ${briefErr(lp)}`);

  const lb = await chatTolerant(model, {
    messages: [{ role: "user", content: "Say hi." }],
    logit_bias: { "0": -10 }, max_tokens: 256, reasoning_effort: "low",
  });
  record(model, "P4.logit-bias", lb.status === 200,
    lb.status === 200 ? "accepted" : `rejected: ${briefErr(lb)}`);
}

// P5: vision
async function pVision(model, isKimi) {
  const mk = (name) => `data:image/png;base64,${solidPng(32, COLORS[name]).toString("base64")}`;
  if (!isKimi) {
    const r = await chat(model, {
      messages: [{ role: "user", content: [
        { type: "text", text: "What color is this image? One word." },
        { type: "image_url", image_url: { url: mk("red") } },
      ] }],
      max_tokens: 256, reasoning_effort: "low",
    });
    const graceful = r.status >= 400 && r.status < 500;
    return record(model, "P5.vision-reject", graceful,
      graceful ? `gracefully rejected image input (HTTP ${r.status})` : r.status === 200 ? "accepted image despite text-only catalogue!" : `5xx on image input (bug): ${briefErr(r)}`);
  }
  const single = await chat(model, {
    messages: [{ role: "user", content: [
      { type: "text", text: "What single color dominates this image? Reply with only the color name." },
      { type: "image_url", image_url: { url: mk("red") } },
    ] }],
    max_tokens: 1024, reasoning_effort: "low",
  });
  const sc = msgOf(single)?.content ?? "";
  record(model, "P5.vision-single", single.status === 200 && /red/i.test(sc),
    single.status === 200 ? `answered: ${sc.slice(0, 40)}` : `rejected/failed: ${briefErr(single)}`);

  const seq = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange", "purple"];
  const images = Array.from({ length: 3 }, () => seq).flat(); // 24
  const burst = await chat(model, {
    messages: [{ role: "user", content: [
      { type: "text", text: `I attached ${images.length} images in order. Each is a solid color. Reply with ONLY a JSON array of ${images.length} color names in order (e.g. ["red","green",...]).` },
      ...images.map((n) => ({ type: "image_url", image_url: { url: mk(n) } })),
    ] }],
    max_tokens: 4096, reasoning_effort: "low",
  });
  if (burst.status !== 200) return record(model, "P5.vision-burst24", false, `24-image burst rejected: ${briefErr(burst)}`);
  const bc = msgOf(burst)?.content ?? "";
  let arr = null;
  try { arr = JSON.parse(bc.replace(/^[^\[]*/, "").replace(/[^\]]*$/, "")); } catch {}
  if (!Array.isArray(arr)) return record(model, "P5.vision-burst24", false, `no JSON array in reply: ${bc.slice(0, 80)}`);
  const correct = arr.slice(0, 24).filter((v, i) => String(v).toLowerCase().trim() === images[i]).length;
  record(model, "P5.vision-burst24", correct >= 20, `${arr.length} colors returned, ${correct}/24 correct`);
}

async function runModel(model) {
  console.log(`\n===== ${model} =====`);
  await pReasoning(model);
  await pToolsPreserved(model);
  await pGrammar(model);
  await pSampling(model);
  await pVision(model, /kimi/i.test(model));
}

for (const m of MODELS) await runModel(m);

const dir = path.join(path.dirname(fileURLToPath(import.meta.url)), "probe-results");
fs.mkdirSync(dir, { recursive: true });
fs.writeFileSync(path.join(dir, `probe-${Date.now()}.json`), JSON.stringify(results, null, 2));

const fails = results.filter((r) => r.pass === false);
console.log(`\n===== SUMMARY: ${results.length - fails.length}/${results.length} passed =====`);
for (const f of fails) console.log(`FAIL  ${f.model}  ${f.probe}  ${f.note}`);
process.exit(fails.length ? 1 : 0);
