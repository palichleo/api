// rag/embedder.js
const path = require('path');
const os = require('os');

/* 1) Forcer UN SEUL onnxruntime-web global et SANS worker */
globalThis.ort = require('onnxruntime-web');
const ort = globalThis.ort;
ort.env.wasm.proxy = false; // pas de Worker -> évite ERR_WORKER_PATH
try {
  // pointer vers les wasm packagés (évite la résolution "blob:nodedata")
  ort.env.wasm.wasmPaths = path.dirname(require.resolve('onnxruntime-web/dist/ort-wasm.wasm'));
} catch (_) {}
ort.env.wasm.numThreads = Math.max(1, Math.min(8, os.cpus().length));

/* 2) Importer Transformers APRES la config ORT globale */
const { pipeline, env } = require('@xenova/transformers');
env.allowRemoteModels = true;
env.localModelPath = '/opt/models';
// ceinture et bretelles : pas de proxy côté config Transformers non plus
try { env.backends.onnx.wasm.proxy = false; } catch (_) {}

/* 3) Embedding "qualité" (lourd). Mets EMBED_QUANTIZED=1 si ça rame trop */
const MODEL_ID = process.env.EMBED_MODEL || 'Xenova/bge-m3';
const EMBED_QUANT = process.env.EMBED_QUANTIZED === '1';

let embedder;
async function getEmbedder() {
  if (!embedder) {
    embedder = await pipeline('feature-extraction', MODEL_ID, { quantized: EMBED_QUANT });
  }
  return embedder;
}

async function embed(text) {
  const m = await getEmbedder();
  const out = await m(text, { pooling: 'mean', normalize: true });
  if (Array.isArray(out)) return Array.from(out[0].data);
  return Array.from(out.data);
}

async function embedMany(texts) {
  const m = await getEmbedder();
  const out = await m(texts, { pooling: 'mean', normalize: true });
  if (Array.isArray(out)) return out.map(x => Array.from(x.data));

  const data = out.data;
  const dims = out.dims || [];
  if (!dims.length) return [Array.from(data)];

  let batch, dim;
  if (dims.length === 2) [batch, dim] = dims;
  else if (dims.length === 1) { batch = texts.length; dim = dims[0]; }
  else { batch = texts.length; dim = data.length / batch; }

  const res = [];
  for (let i = 0; i < batch; i++) {
    const start = i * dim;
    res.push(Array.from(data.slice(start, start + dim)));
  }
  return res;
}

module.exports = { embed, embedMany };
