// rag/embedder.js
const { pipeline, env } = require('@xenova/transformers');
const os = require('os');

// ⚡ optimisations CPU sûres
env.backends.onnx.wasm.numThreads = Math.max(1, Math.min(4, os.cpus().length));
env.allowRemoteModels = true;              // repasse à false après le 1er run si tu poses les modèles dans /opt/models
env.localModelPath = '/opt/models';

// ✅ modèle rapide multilingue (bien plus vite que bge-m3 sur CPU)
//   change via EMBED_MODEL si besoin
const MODEL_ID = process.env.EMBED_MODEL || 'Xenova/paraphrase-multilingual-MiniLM-L12-v2';

let embedder;
async function getEmbedder() {
  if (!embedder) {
    embedder = await pipeline('feature-extraction', MODEL_ID, { quantized: true });
  }
  return embedder;
}

async function embed(text) {
  const m = await getEmbedder();
  const out = await m(text, { pooling: 'mean', normalize: true });
  if (Array.isArray(out)) return Array.from(out[0].data);
  return Array.from(out.data);
}

// ✅ batching robuste (gère Tensor batched ou array de Tensors)
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
