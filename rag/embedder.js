// rag/embedder.js
const { pipeline, env } = require('@xenova/transformers');

// ⚡ petites optimisations CPU sûres
env.backends.onnx.wasm.numThreads = Math.max(1, Math.min(4, require('os').cpus().length));
env.allowRemoteModels = true;                 // tu peux remettre false après le premier run
env.localModelPath = '/opt/models';

const MODEL_ID = process.env.EMBED_MODEL || 'Xenova/bge-m3'; // garde ton choix actuel

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
  return Array.from(out.data);
}

async function embedMany(texts) {
  const m = await getEmbedder();
  const outs = await m(texts, { pooling: 'mean', normalize: true });
  return outs.map(x => Array.from(x.data));
}

module.exports = { embed, embedMany };
