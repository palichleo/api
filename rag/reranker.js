// rag/reranker.js
const { pipeline, env } = require('@xenova/transformers');
env.allowRemoteModels = false;
env.localModelPath = '/opt/models';

const USER_MODEL = process.env.RERANK_MODEL || '';
const CANDIDATES = [
  USER_MODEL,
  'Xenova/bge-reranker-base',                 // PRIORITAIRE (CPU OK)
  'Xenova/ms-marco-MiniLM-L-6-v2'            // fallback
].filter(Boolean);

const MAX_CHARS = parseInt(process.env.RERANK_TRUNC || '900', 10);
const BATCH     = parseInt(process.env.RERANK_BATCH || '8', 10);

let reranker;

async function getReranker() {
  if (reranker) return reranker;
  let lastErr;
  for (const id of CANDIDATES) {
    try {
      console.log('[RERANK] trying model:', id);
      reranker = await pipeline('text-classification', id, { quantized: true });
      console.log('[RERANK] loaded model:', id);
      return reranker;
    } catch (e) {
      console.warn('[RERANK] failed:', id, '-', e.message);
      lastErr = e;
    }
  }
  throw new Error(`No rerank model available. Tried: ${CANDIDATES.join(', ')} | last error: ${lastErr?.message}`);
}

async function rerankCrossEncoder(query, candidates) {
  const ranker = await getReranker();
  const q = String(query ?? '');
  const scored = [];

  // batching
  for (let i = 0; i < candidates.length; i += BATCH) {
    const batchTexts = candidates.slice(i, i + BATCH).map(c => {
      const d = String(c.text ?? '');
      const qT = q.slice(0, MAX_CHARS);
      const dT = d.length > MAX_CHARS ? (d.slice(0, MAX_CHARS) + '…') : d;
      
      // ✅ Format STRING pour cross-encoder (pas array)
      return `Query: ${qT} Document: ${dT}`;
    });

    // ✅ Passe des strings, pas des arrays
    const out = await ranker(batchTexts);
    
    // Handle les différents formats de sortie
    const results = Array.isArray(out) ? out : [out];

    for (let j = 0; j < results.length && (i + j) < candidates.length; j++) {
      const result = results[j];
      // Extrait le score (peut être .score ou .LABEL_1 selon le modèle)
      const score = result?.score ?? result?.LABEL_1 ?? 0;
      scored.push({ ...candidates[i + j], rerank: score });
    }
  }

  scored.sort((a, b) => b.rerank - a.rerank);
  return scored;
}

module.exports = { rerankCrossEncoder };