// rag/reranker.js
const { pipeline, env } = require('@xenova/transformers');

// Autoriser le téléchargement de modèles (CPU only)
env.allowRemoteModels = true;

// Changeable via variable d'env, sinon par défaut MiniLM
const MODEL_ID = process.env.RERANK_MODEL || 'cross-encoder/ms-marco-MiniLM-L-6-v2';

let reranker;

/** Lazy init du pipeline cross-encoder, en quantifié (léger CPU) */
async function getReranker() {
  if (!reranker) {
    reranker = await pipeline('text-classification', MODEL_ID, { quantized: true });
  }
  return reranker;
}

/**
 * Rerank par cross-encoder : prend (query, [{text, ...}]) et renvoie les mêmes candidats triés
 */
async function rerankCrossEncoder(query, candidates) {
  const ranker = await getReranker();

  const scored = [];
  for (const c of candidates) {
    // Le cross-encoder lit la paire (query, passage)
    const out = await ranker({ text: query, text_pair: c.text }, { topk: 1 });
    const score = Array.isArray(out) && out[0]?.score != null ? out[0].score : 0;
    scored.push({ ...c, rerank: score });
  }
  scored.sort((a, b) => b.rerank - a.rerank);
  return scored;
}

module.exports = { rerankCrossEncoder };
