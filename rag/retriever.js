// rag/retriever.js

const { ChromaClient } = require('chromadb');
const { embed } = require('./embedder');
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
        embeddingFunction: null
      });
    }
  }
  return collection;
}

async function addDocument(text, source = 'unknown') {
  const col = await initCollection();
  const embedding = await embed(text); // tableau 1D
  await col.add({
    ids: [crypto.randomUUID()],
    documents: [text],
    metadatas: [{ source }],
    embeddings: [embedding]
  });
}

// --- utils ---
function l2norm(v) {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}
function cosine(a, b) {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  for (let i = 0; i < n; i++) dot += a[i] * b[i];
  const na = l2norm(a) || 1e-9;
  const nb = l2norm(b) || 1e-9;
  return dot / (na * nb);
}

// Maximal Marginal Relevance (diversité)
function mmr(queryVec, candVecs, k = 3, lambda = 0.5) {
  const selected = [];
  const remaining = new Set(candVecs.map((_, i) => i));

  // Similarité à la requête
  const simToQuery = candVecs.map(v => cosine(queryVec, v));

  while (selected.length < Math.min(k, candVecs.length)) {
    let bestI = null;
    let bestScore = -Infinity;

    for (const i of remaining) {
      let maxRedundancy = 0;
      if (selected.length) {
        for (const j of selected) {
          const s = cosine(candVecs[i], candVecs[j]);
          if (s > maxRedundancy) maxRedundancy = s;
        }
      }
      const score = lambda * simToQuery[i] - (1 - lambda) * maxRedundancy;
      if (score > bestScore) {
        bestScore = score;
        bestI = i;
      }
    }

    selected.push(bestI);
    remaining.delete(bestI);
  }
  return selected;
}

// Rerank lexical très léger (compte mots requis présent dans le chunk)
function lexicalScore(query, text) {
  const q = query.toLowerCase().split(/[^a-zàâçéèêëîïôûùüÿñæœ0-9]+/).filter(Boolean);
  const t = text.toLowerCase();
  let s = 0;
  for (const w of q) {
    if (w.length >= 3 && t.includes(w)) s += 1;
  }
  return s;
}

async function retrieveRelevant(query, kFinal = 5, kInitial = 20) {
  const col = await initCollection();
  const queryEmbedding = await embed(query);

  // 1) récupère large
  const results = await col.query({
    queryEmbeddings: [queryEmbedding],
    nResults: kInitial
  });

  const docs = results.documents[0] || [];
  const metas = results.metadatas[0] || [];
  const dists = results.distances[0] || [];

  if (docs.length === 0) return [];

  // 2) ré‑encode localement les candidats pour MMR (coût OK pour ~12)
  const candEmbeddings = [];
  for (const text of docs) {
    candEmbeddings.push(await embed(text));
  }

  // 3) MMR pour diversité
  const mmrIdx = mmr(queryEmbedding, candEmbeddings, Math.min(kFinal * 2, docs.length), 0.5);

  // 4) Rerank lexical simple sur le sous‑ensemble MMR
  const scored = mmrIdx.map(i => {
    const cosSim = cosine(queryEmbedding, candEmbeddings[i]); // [-1..1]
    const lex = lexicalScore(query, docs[i]);                 // >=0
    const score = 0.8 * cosSim + 0.2 * (Math.min(lex, 10) / 10); // mixture
    return {
      text: docs[i],
      source: metas[i]?.source || 'inconnu',
      // on convertit distance → sim si jamais tu veux comparer : sim ≈ -dist (approx pour Chroma cosine)
      rawDistance: dists[i],
      score
    };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, kFinal);
}

module.exports = { addDocument, retrieveRelevant };
