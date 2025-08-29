// rag/retriever.js
const crypto = require('crypto');
const { ChromaClient } = require('chromadb');
const { embed, embedMany } = require('./embedder');

const client = new ChromaClient({ host: 'localhost', port: 8000, ssl: false });
let collection = null;

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

// ---------------- utils
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function norm(a) { return Math.sqrt(dot(a, a)); }
function cosine(a, b) { return dot(a, b) / (norm(a) * norm(b) + 1e-10); }

// Stopwords FR (petit set suffisant pour nos requêtes)
const STOP = new Set('je tu il elle nous vous ils elles de du des le la les un une et ou en au aux sur pour par avec sans sous chez dans que qui quoi dont où est sont été étais était êtres avoir ai as a avons avez ont plus moins très ne pas ni mais donc or car'.split(' '));
function normalize(str) {
  return (str || '')
    .toLowerCase()
    .normalize('NFKD').replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9\s\-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}
function tokens(str) {
  return normalize(str).split(' ').filter(t => t && !STOP.has(t));
}
function jaccard(a, b) {
  const A = new Set(a), B = new Set(b);
  let inter = 0; for (const x of A) if (B.has(x)) inter++;
  const uni = A.size + B.size - inter;
  return uni ? inter / uni : 0;
}
function extractYear(text) {
  const m = (text || '').match(/\b(19|20)\d{2}\b/);
  return m ? parseInt(m[0], 10) : null;
}

// ---------------- MMR (diversité)
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

// ---------------- Query expansion LOCAL (sans LLM)
function expandQueriesLocal(query, n = 3) {
  const q = normalize(query);
  const qs = new Set([q]);

  // 1) version "keywords" (sans stopwords)
  const toks = tokens(query);
  if (toks.length) qs.add(toks.join(' '));

  // 2) synonymes simples FR ciblés pour ton domaine
  const syn = [
    ['parcours','experience','cv','formation'],
    ['projet','realisations','travaux','portfolio'],
    ['intelligence artificielle','ia','machine learning','ml'],
    ['traitement de donnees','analyse de donnees','data','statistiques'],
    ['freelance','independant','consultant']
  ];
  syn.forEach(group => {
    for (const w of group) {
      if (q.includes(w)) group.forEach(v => qs.add(q.replace(w, v)));
    }
  });

  // 3) variantes d’ordre (bigrams)
  for (let i = 0; i + 1 < toks.length; i++) {
    qs.add(`${toks[i+1]} ${toks[i]}`);
  }

  return Array.from(qs).slice(0, Math.max(2, n + 1)); // original + n variantes
}

// ---------------- add / bulk add (embeddings fournis par nous)
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

/**
 * retrieveRelevant(query, kFinal=6, kInitial=96)
 * - Multi-Query LOCAL -> union candidats
 * - Rerank local (cosine + overlap + recence)
 * - MMR -> diversité
 */
async function retrieveRelevant(query, kFinal = 6, kInitial = 96, dbg = {}) {
  const col = await initCollection();

  // 1) expansions locales
  const queries = expandQueriesLocal(query, 4);
  const qMainTokens = tokens(query);

  // 2) embed des requêtes
  const qEmbs = await embedMany(queries);

  // 3) récup pool large par requête
  const candidates = new Map();
  for (const qEmb of qEmbs) {
    const res = await col.query({
      queryEmbeddings: [qEmb],
      nResults: kInitial,
      include: ['documents', 'metadatas', 'distances', 'embeddings'],
    });
    const ids = res.ids?.[0] || [];
    const docs = res.documents?.[0] || [];
    const metas = res.metadatas?.[0] || [];
    const embs  = res.embeddings?.[0] || [];
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i] || crypto.randomUUID();
      if (!candidates.has(id)) {
        candidates.set(id, {
          id,
          text: docs[i],
          source: metas[i]?.source || 'unknown',
          emb: embs[i],
        });
      }
    }
  }
  const pool = Array.from(candidates.values());
  if (pool.length === 0) return [];

  // 4) Rerank LOCAL (cosine + jaccard + recency)
  const yearNow = new Date().getFullYear();
  const qEmbMain = qEmbs[0];
  const ranked = pool.map((d) => {
    const cos = cosine(qEmbMain, d.emb);
    const over = jaccard(qMainTokens, tokens(d.text));
    const y = extractYear(d.text);
    const rec = y ? Math.max(0, 1 - Math.min(10, Math.abs(yearNow - y)) / 10) : 0.5; // proche de maintenant → score↑
    const score = 0.7 * cos + 0.2 * over + 0.1 * rec;
    return { ...d, score };
  }).sort((a, b) => b.score - a.score);

  // 5) MMR pour diversité sur un top large
  const topForMMR = ranked.slice(0, Math.max(kFinal * 4, 24));
  const embs = topForMMR.map(x => x.emb);
  const idxs = mmr(qEmbMain, embs, 0.65, Math.min(kFinal * 2, topForMMR.length));
  const diverse = idxs.map(i => topForMMR[i]);

  // 6) tri final par score puis coupe
  const final = diverse.sort((a, b) => b.score - a.score).slice(0, kFinal);

  return final.map((x, i) => ({
    i,
    source: x.source,
    score: +x.score.toFixed(4),
    text: x.text
  }));
}

module.exports = { addDocument, addDocuments, retrieveRelevant };
