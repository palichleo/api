const { pipeline, env } = require('@xenova/transformers');
env.allowRemoteModels = true;
const MODEL_ID = process.env.RERANK_MODEL || 'cross-encoder/bge-reranker-base';
const MAX_CHARS = parseInt(process.env.RERANK_TRUNC || '1200', 10); // ~500-600 tokens
const BATCH = parseInt(process.env.RERANK_BATCH || '8', 10);

let reranker;
async function getReranker() {
  if (!reranker) {
    reranker = await pipeline('text-classification', MODEL_ID, { quantized: true });
  }
  return reranker;
}

async function rerankCrossEncoder(query, candidates) {
  const ranker = await getReranker();
  const scored = [];
  for (let i = 0; i < candidates.length; i += BATCH) {
    const batch = candidates.slice(i, i + BATCH).map(c => ({
      text: query.slice(0, MAX_CHARS),
      text_pair: c.text.length > MAX_CHARS ? (c.text.slice(0, MAX_CHARS) + '…') : c.text
    }));
    const out = await ranker(batch, { topk: 1 }); // batch inference
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