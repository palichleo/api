// rag/embedder.js
const ort = require('onnxruntime-web'); // ← IMPORTANT: avant transformers
ort.env.wasm.proxy = false;

const { pipeline, env } = require('@xenova/transformers');
const os = require('os');

env.backends.onnx.wasm.numThreads = Math.max(1, Math.min(4, os.cpus().length));
env.backends.onnx.wasm.proxy = false;   // ceinture + bretelles
env.allowRemoteModels = true;
env.localModelPath = process.env.XENOVA_CACHE || '/opt/models';

const MODEL_ID = process.env.EMBED_MODEL || 'Xenova/multilingual-e5-base';

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
