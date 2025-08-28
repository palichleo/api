// rag/embedder.js
const { pipeline, env } = require('@xenova/transformers');
const os = require('os');
const path = require('path');

// ====== IMPORTANT: forcer le backend wasm (sans worker) ======
env.backends.onnx.wasm.proxy = false; // évite l'erreur ERR_WORKER_PATH
env.backends.onnx.wasm.numThreads = Math.max(1, Math.min(8, os.cpus().length));
// (optionnel) pointer explicitement vers les wasm locaux
try {
  env.backends.onnx.wasm.wasmPaths = path.dirname(require.resolve('onnxruntime-web/dist/ort-wasm.wasm'));
} catch (_) { /* ignore si introuvable */ }

// Modèles locaux/remote
env.allowRemoteModels = true;
env.localModelPath = '/opt/models';

// Embedding "qualité" (lourd mais précis)
const MODEL_ID = process.env.EMBED_MODEL || 'Xenova/bge-m3';
const EMBED_QUANT = process.env.EMBED_QUANTIZED === '1'; // 0 = qualité, 1 = plus rapide

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
