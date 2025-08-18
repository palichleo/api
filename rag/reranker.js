// rag/reranker.js
const { pipeline, env } = require('@xenova/transformers');
env.allowRemoteModels = true;                 // ou false si tu mets le modèle en local
// env.localModelPath = '/opt/models';       // décommente si tu clones en local

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
  const q = (query ?? '').toString();         // <- évite "text.split is not a function"
  const scored = [];

  for (let i = 0; i < candidates.length; i += BATCH) {
    const batch = candidates.slice(i, i + BATCH).map(c => {
      const d = (c.text ?? '').toString();
      return {
        text:      q.slice(0, MAX_CHARS),
        text_pair: d.length > MAX_CHARS ? (d.slice(0, MAX_CHARS) + '…') : d,
      };
    });

    const out = await ranker(batch, { topk: 1 });
    const arr = Array.isArray(out[0]) ? out : out.map(x => [x]);
    for (let j = 0; j < arr.length; j++) {
      const score = arr[j][0]?.score ?? 0;
      scored.push({ ...candidates[i + j], rerank: score });
    }
  }

  scored.sort((a, b) => b.rerank - a.rerank);
  return scored;
}

module.exports = { rerankCrossEncoder };
