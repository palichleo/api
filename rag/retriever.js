// rag/retriever.js
const { ChromaClient } = require('chromadb');
const { embed } = require('./embedder');
const { rerankCrossEncoder } = require('./reranker');
const crypto = require('crypto');

const chroma = new ChromaClient({ host: 'localhost', port: 8000 });

let collection;
async function initCollection() {
  if (!collection) {
    try {
      collection = await chroma.getCollection({ name: 'leoknowledge' });
    } catch {
      collection = await chroma.createCollection({
        name: 'leoknowledge',
        embeddingFunction: null // on fournit nous-mêmes les embeddings à add/query
      });
    }
  }
  return collection;
}

async function addDocument(text, source = 'unknown') {
  const col = await initCollection();
  const embedding = await embed(text);
  await col.add({
    ids: [crypto.randomUUID()],
    documents: [text],
    metadatas: [{ source }],
    embeddings: [embedding]
  });
}

// --- utils ---
function l2norm(v) { return Math.sqrt(v.reduce((s, x) => s + x * x, 0)); }
function cosine(a, b) {
  const n = Math.min(a.length, b.length);
  let dot = 0; for (let i = 0; i < n; i++) dot += a[i] * b[i];
  const na = l2norm(a) || 1e-9, nb = l2norm(b) || 1e-9;
  return dot / (na * nb);
}

// MMR (diversité)
function mmr(queryVec, candVecs, k = 3, lambda = 0.5) {
  const selected = [], remaining = new Set(candVecs.map((_, i) => i));
  const simToQuery = candVecs.map(v => cosine(queryVec, v));
  while (selected.length < Math.min(k, candVecs.length)) {
    let bestI = null, bestScore = -Infinity;
    for (const i of remaining) {
      let maxRed = 0;
      for (const j of selected) maxRed = Math.max(maxRed, cosine(candVecs[i], candVecs[j]));
      const score = lambda * simToQuery[i] - (1 - lambda) * maxRed;
      if (score > bestScore) { bestScore = score; bestI = i; }
    }
    selected.push(bestI); remaining.delete(bestI);
  }
  return selected;
}

// Fallback lexical très léger
function lexicalScore(query, text) {
  const q = query.toLowerCase().split(/[^a-zàâçéèêëîïôûùüÿñæœ0-9]+/).filter(Boolean);
  const t = text.toLowerCase(); let s = 0;
  for (const w of q) if (w.length >= 3 && t.includes(w)) s += 1;
  return s;
}

async function retrieveRelevant(query, kFinal = 5, kInitial = 20) {
  const col = await initCollection();
  const queryEmbedding = await embed(query);

  // 1) Récup large depuis Chroma
  const results = await col.query({
    queryEmbeddings: [queryEmbedding],
    nResults: kInitial
  });

  const docs  = results.documents[0] || [];
  const metas = results.metadatas[0] || [];
  const dists = results.distances[0] || [];
  if (docs.length === 0) return [];

  // 2) Ré-encode local pour MMR (kInitial petit -> OK CPU)
  const candEmbeddings = [];
  for (const text of docs) candEmbeddings.push(await embed(text));

  // 3) MMR pour diversité (on garde ~2×kFinal)
  const mmrIdx = mmr(queryEmbedding, candEmbeddings, Math.min(kFinal * 2, docs.length), 0.5);

  // 4) Sous-ensemble candidats
  const subset = mmrIdx.map(i => ({
    text: docs[i],
    source: metas[i]?.source || 'inconnu',
    rawDistance: dists[i],
    cosSim: cosine(queryEmbedding, candEmbeddings[i])
  }));

  // 5) Rerank cross-encoder (CPU). Fallback lexical si indispo.
  let reranked;
  try {
    reranked = await rerankCrossEncoder(query, subset);
  } catch (e) {
    console.warn('[RERANK] cross-encoder indisponible, fallback lexical:', e.message);
    reranked = subset.map(c => ({
      ...c,
      rerank: 0.8 * c.cosSim + 0.2 * (Math.min(lexicalScore(query, c.text), 10) / 10)
    })).sort((a, b) => b.rerank - a.rerank);
  }

  // 6) Top-kFinal (on renvoie "score" = score rerank)
  return reranked.slice(0, kFinal).map(c => ({
    text: c.text,
    source: c.source,
    rawDistance: c.rawDistance,
    score: c.rerank
  }));
}

module.exports = { addDocument, retrieveRelevant };
