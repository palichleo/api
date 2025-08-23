// rag/embedder.js
const { pipeline, env } = require('@xenova/transformers');

// Petites optimisations CPU (sûres)
env.backends.onnx.wasm.numThreads = Math.max(1, Math.min(4, require('os').cpus().length));
env.allowRemoteModels = true;              // tu peux remettre false après le 1er run
env.localModelPath = '/opt/models';

const MODEL_ID = process.env.EMBED_MODEL || 'Xenova/bge-m3';

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
  // out est un Tensor { data, dims } ou parfois { data } simple
  if (Array.isArray(out)) {
    // rare selon version; on normalise
    return Array.from(out[0].data);
  }
  return Array.from(out.data);
}

// ✅ gestion de toutes les formes de sortie (array ou Tensor batched)
async function embedMany(texts) {
  const m = await getEmbedder();
  const out = await m(texts, { pooling: 'mean', normalize: true });

  if (Array.isArray(out)) {
    // Certains pipelines renvoient un tableau de tensors
    return out.map(x => Array.from(x.data));
  }

  // Cas standard: un unique Tensor batched
  const data = out.data;        // Float32Array
  const dims = out.dims || [];  // p.ex. [batch, hidden] après pooling
  if (!dims.length) {
    // Pas de dims -> on considère un seul vecteur
    return [Array.from(data)];
  }

  let batch, dim;
  if (dims.length === 2) {
    [batch, dim] = dims;
  } else if (dims.length === 1) {
    // Certains backends ne gardent qu'une dim après pooling
    batch = texts.length;
    dim = dims[0];
  } else {
    // fallback
    batch = texts.length;
    dim = data.length / batch;
  }

  const res = [];
  for (let i = 0; i < batch; i++) {
    const start = i * dim;
    res.push(Array.from(data.slice(start, start + dim)));
  }
  return res;
}

module.exports = { embed, embedMany };
