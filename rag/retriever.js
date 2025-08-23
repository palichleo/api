// rag/retriever.js
const crypto = require('crypto');
const { ChromaClient } = require('chromadb');
const { embed, embedMany } = require('./embedder');

const client = new ChromaClient({ path: 'http://localhost:8000' });
let collection = null;

async function initCollection() {
  if (collection) return collection;
  const name = 'leoknowledge';
  const list = await client.listCollections();
  const found = list.find(c => c.name === name);
  if (found) {
    collection = await client.getCollection({ name, embeddingFunction: null });
  } else {
    // ✅ forcer l’espace cosine côté index (cohérent avec tes vecteurs normalisés)
    collection = await client.createCollection({
      name,
      embeddingFunction: null,
      metadata: { 'hnsw:space': 'cosine' }
    });
  }
  return collection;
}

async function addDocument(text, source = 'unknown') {
  const col = await initCollection();
  const e = await embed(text);
  await col.add({
    ids: [crypto.randomUUID()],
    documents: [text],
    metadatas: [{ source }],
    embeddings: [e],
  });
}

async function addDocuments(items) {
  if (!items || items.length === 0) return;
  const col = await initCollection();
  const texts   = items.map(x => x.text);
  const sources = items.map(x => x.source || 'unknown');
  const embs    = await embedMany(texts);
  const ids     = items.map(() => crypto.randomUUID());
  await col.add({
    ids,
    documents: texts,
    metadatas: sources.map(s => ({ source: s })),
    embeddings: embs,
  });
}

// --- utils cosine/MMR (rapides)
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function norm(a) { return Math.sqrt(dot(a, a)); }
function cosine(a, b) { return dot(a, b) / (norm(a) * norm(b) + 1e-10); }

function mmr(queryEmb, candEmbs, lambda = 0.65, topK = 6) {
  const n = candEmbs.length;
  const selected = [];
  const used = new Set();
  const simToQuery = candEmbs.map(e => cosine(queryEmb, e));

  while (selected.length < Math.min(topK, n)) {
    let best = -1, bestScore = -Infinity;
    for (let i = 0; i < n; i++) {
      if (used.has(i)) continue;
      let div = 0;
      for (const j of selected) div = Math.max(div, cosine(candEmbs[i], candEmbs[j]));
      const score = lambda * simToQuery[i] - (1 - lambda) * div;
      if (score > bestScore) { bestScore = score; best = i; }
    }
    used.add(best);
    selected.push(best);
  }
  return selected;
}

async function retrieveRelevant(query, kFinal = 3, kInitial = 12) {
  const col = await initCollection();
  const qEmb = await embed(query);

  const res = await col.query({
    queryEmbeddings: [qEmb],
    nResults: kInitial,
    include: ['documents', 'metadatas', 'distances', 'embeddings'],
  });

  const docs = res.documents?.[0] || [];
  const metas = res.metadatas?.[0] || [];
  const dists = res.distances?.[0] || [];
  const embs  = res.embeddings?.[0] || [];

  if (!docs.length) return [];

  // Tri initial par similarité cosine (plus stable que 'distance' renvoyée)
  const sims = embs.map(e => cosine(qEmb, e));
  const order = sims
    .map((s, i) => ({ i, s }))
    .sort((a, b) => b.s - a.s)
    .map(o => o.i);

  const orderedDocs  = order.map(i => docs[i]);
  const orderedMetas = order.map(i => metas[i]);
  const orderedEmbs  = order.map(i => embs[i]);

  // Diversification MMR puis top-kFinal
  const mmrIdxs = mmr(qEmb, orderedEmbs, 0.65, Math.min(kFinal * 2, orderedDocs.length));
  const subset = mmrIdxs.map(i => ({
    text: orderedDocs[i],
    source: orderedMetas[i]?.source || 'unknown',
    cosSim: cosine(qEmb, orderedEmbs[i]),
  }));

  // ✅ pas de rerank (cross-encoder) — vitesse maximale CPU
  const top = subset
    .sort((a, b) => b.cosSim - a.cosSim)
    .slice(0, kFinal)
    .map((x, i) => ({ i, source: x.source, score: +x.cosSim.toFixed(4), text: x.text }));

  return top;
}

module.exports = { addDocument, addDocuments, retrieveRelevant };
