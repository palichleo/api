// rag/embedder.js
const { pipeline } = require('@xenova/transformers');
const fs = require('fs');

let embedder;
async function initEmbedder() {
  if (!embedder) {
    embedder = await pipeline('feature-extraction', 'Xenova/bge-m3');
  }
  return embedder;
}

async function embed(text) {
  const model = await initEmbedder();
  const output = await model(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

module.exports = { embed };