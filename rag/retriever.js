// rag/retriever.js
const crypto = require('crypto');
const { ChromaClient } = require('chromadb');
const { embed, embedMany } = require('./embedder');
const { pipeline } = require('@xenova/transformers');

const client = new ChromaClient({ host: 'localhost', port: 8000, ssl: false });
let collection = null;
let reranker = null;

async function initCollection() {
  if (collection) return collection;
  const name = 'leoknowledge';
  try {
    collection = await client.getCollection({ name, embeddingFunction: null });
  } catch {
    collection = await client.createCollection({
      name,
      embeddingFunction: null,
      metadata: { 'hnsw:space': 'cosine' }
    });
  }
  return collection;
}

// --- utils cosines / mmr
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

// --- add / bulk add (inchangé)
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

// --- Multi-Query via Groq (paraphrases)
async function expandQueriesLLM(query, n = 3) {
  const key = process.env.GROQ_API_KEY;
  const model = process.env.GROQ_MODEL || 'llama-3.1-70b-versatile';
  if (!key) return [query];
  const url = 'https://api.groq.com/openai/v1/chat/completions';
  const r = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${key}`
    },
    body: JSON.stringify({
      model,
      temperature: 0.2,
      max_tokens: 180,
      stream: false,
      messages: [
        { role: 'system', content: 'Réécris la question en variations courtes, différentes formulations. Donne N lignes, une paraphrase par ligne, sans numérotation.' },
        { role: 'user', content: `N=${n}\nQuestion: ${query}` }
      ]
    })
  });
  if (!r.ok) return [query];
  const data = await r.json();
  const txt = data.choices?.[0]?.message?.content || '';
  const lines = txt.split('\n').map(s => s.trim()).filter(Boolean);
  const out = Array.from(new Set([query, ...lines]));
  return out.slice(0, n + 1);
}

// --- Cross-encoder reranker
async function getReranker() {
  if (!reranker) {
    const id = process.env.RERANK_MODEL || 'Xenova/bge-reranker-base';
    reranker = await pipeline('text-classification', id, { quantized: false });
  }
  return reranker;
}

async function crossEncodeScore(query, doc) {
  const r = await (await getReranker())({ text: query, text_pair: doc });
  // r: [{label: 'LABEL_1', score: 0.87}, ...]
  return Array.isArray(r) ? (r[0]?.score ?? 0) : (r?.score ?? 0);
}

/**
 * retrieveRelevant(query, kFinal=6, kInitial=64)
 * - Multi-Query LLM -> union des candidats
 * - Rerank cross-encoder -> tri qualité
 * - MMR -> diversité finale
 */
async function retrieveRelevant(query, kFinal = 6, kInitial = 64, dbg = {}) {
  const col = await initCollection();

  // 1) expansions de requêtes
  const queries = await expandQueriesLLM(query, 3);

  // 2) embed des requêtes
  const qEmbs = await embedMany(queries);

  // 3) récup candidats par requête (pool large)
  const candidates = new Map(); // id -> {text, source, emb?, dist?, scores...}
  for (const qEmb of qEmbs) {
    const res = await col.query({
      queryEmbeddings: [qEmb],
      nResults: kInitial,
      include: ['ids', 'documents', 'metadatas', 'distances', 'embeddings'],
    });
    const ids = res.ids?.[0] || [];
    const docs = res.documents?.[0] || [];
    const metas = res.metadatas?.[0] || [];
    const dists = res.distances?.[0] || [];
    const embs  = res.embeddings?.[0] || [];
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i] || crypto.randomUUID();
      if (!candidates.has(id)) {
        candidates.set(id, {
          id,
          text: docs[i],
          source: metas[i]?.source || 'unknown',
          emb: embs[i],
          dist: dists[i],
        });
      }
    }
  }
  const pool = Array.from(candidates.values());
  if (pool.length === 0) return [];

  // 4) cross-encoder rerank (⚠️ lourd mais précis)
  //    On score sur la question d'origine (pas les paraphrases) pour éviter les biais.
  for (const c of pool) {
    c.ce = await crossEncodeScore(query, c.text);
  }
  pool.sort((a, b) => b.ce - a.ce); // meilleur d'abord

  // 5) MMR pour diversité sur le top élargi (ex: top 24)
  const topForMMR = pool.slice(0, Math.max(kFinal * 4, 24));
  const qEmbMain = qEmbs[0];
  const embs = topForMMR.map(x => x.emb);
  const idxs = mmr(qEmbMain, embs, 0.65, Math.min(kFinal * 2, topForMMR.length));
  const diverse = idxs.map(i => topForMMR[i]);

  // 6) tri final par score CE puis coupe à kFinal
  const final = diverse.sort((a, b) => b.ce - a.ce).slice(0, kFinal);

  // sortie attendue par index.js
  return final.map((x, i) => ({
    i,
    source: x.source,
    score: +(x.ce || 0).toFixed(4),
    text: x.text
  }));
}

module.exports = { addDocument, addDocuments, retrieveRelevant };
