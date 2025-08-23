// rag/retriever.js - Version optimisée
const { ChromaClient } = require('chromadb');
const { embed } = require('./embedder');
const { rerankCrossEncoder } = require('./reranker');
const crypto = require('crypto');

const chroma = new ChromaClient({ host: 'localhost', port: 8000 });

let collection;
const embeddingCache = new Map();
const EMBEDDING_CACHE_SIZE = 100;

async function initCollection() {
  if (!collection) {
    try {
      collection = await chroma.getCollection({ name: 'leoknowledge' });
      console.log('[INIT] Collection existante chargée');
    } catch {
      collection = await chroma.createCollection({
        name: 'leoknowledge',
        embeddingFunction: null
      });
      console.log('[INIT] Nouvelle collection créée');
    }
  }
  return collection;
}

// Cache pour les embeddings
async function getCachedEmbedding(text) {
  const cacheKey = text.substring(0, 100); // Clé basée sur le début du texte
  
  if (embeddingCache.has(cacheKey)) {
    console.log('[CACHE] Embedding trouvé dans le cache');
    return embeddingCache.get(cacheKey);
  }
  
  const embedding = await embed(text);
  
  // Gérer la taille du cache
  if (embeddingCache.size >= EMBEDDING_CACHE_SIZE) {
    const firstKey = embeddingCache.keys().next().value;
    embeddingCache.delete(firstKey);
  }
  
  embeddingCache.set(cacheKey, embedding);
  return embedding;
}

async function addDocument(text, source = 'unknown') {
  const col = await initCollection();
  const embedding = await getCachedEmbedding(text);
  await col.add({
    ids: [crypto.randomUUID()],
    documents: [text],
    metadatas: [{ source }],
    embeddings: [embedding]
  });
}

// Calculs vectoriels optimisés
const vectorOps = {
  l2norm: (v) => {
    let sum = 0;
    for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
    return Math.sqrt(sum);
  },
  
  cosine: (a, b) => {
    const n = Math.min(a.length, b.length);
    let dot = 0;
    for (let i = 0; i < n; i++) dot += a[i] * b[i];
    const na = vectorOps.l2norm(a) || 1e-9;
    const nb = vectorOps.l2norm(b) || 1e-9;
    return dot / (na * nb);
  }
};

// MMR optimisé avec early stopping
function mmrOptimized(queryVec, candVecs, k = 3, lambda = 0.6) {
  if (candVecs.length <= k) {
    return candVecs.map((_, i) => i);
  }
  
  const selected = [];
  const remaining = new Set(candVecs.map((_, i) => i));
  
  // Pré-calculer les similarités avec la requête
  const simToQuery = new Float32Array(candVecs.length);
  for (let i = 0; i < candVecs.length; i++) {
    simToQuery[i] = vectorOps.cosine(queryVec, candVecs[i]);
  }
  
  // Sélection MMR
  while (selected.length < k && remaining.size > 0) {
    let bestI = -1;
    let bestScore = -Infinity;
    
    for (const i of remaining) {
      let maxRed = 0;
      
      // Calcul de redondance seulement si nécessaire
      if (selected.length > 0) {
        for (const j of selected) {
          const sim = vectorOps.cosine(candVecs[i], candVecs[j]);
          if (sim > maxRed) maxRed = sim;
        }
      }
      
      const score = lambda * simToQuery[i] - (1 - lambda) * maxRed;
      
      if (score > bestScore) {
        bestScore = score;
        bestI = i;
      }
    }
    
    if (bestI >= 0) {
      selected.push(bestI);
      remaining.delete(bestI);
    }
  }
  
  return selected;
}

// Score lexical optimisé
function lexicalScoreFast(query, text) {
  const queryWords = new Set(
    query.toLowerCase()
      .split(/[^a-zàâçéèêëîïôûùüÿñæœ0-9]+/)
      .filter(w => w.length >= 3)
  );
  
  if (queryWords.size === 0) return 0;
  
  const textLower = text.toLowerCase();
  let score = 0;
  
  for (const word of queryWords) {
    if (textLower.includes(word)) {
      score += 1;
      // Bonus pour les mots exacts
      const regex = new RegExp(`\\b${word}\\b`, 'gi');
      const matches = textLower.match(regex);
      if (matches) score += matches.length * 0.5;
    }
  }
  
  return Math.min(score / queryWords.size, 1);
}

async function retrieveRelevant(query, kFinal = 3, kInitial = 10) {
  console.time('[PERF] Total retrieval');
  
  const col = await initCollection();
  
  // Embedding de la requête avec cache
  console.time('[PERF] Query embedding');
  const queryEmbedding = await getCachedEmbedding(query);
  console.timeEnd('[PERF] Query embedding');
  
  // Recherche ChromaDB
  console.time('[PERF] ChromaDB search');
  const results = await col.query({
    queryEmbeddings: [queryEmbedding],
    nResults: kInitial,
    include: ['documents', 'metadatas', 'distances', 'embeddings']
  });
  console.timeEnd('[PERF] ChromaDB search');
  
  const docs = results.documents[0] || [];
  const metas = results.metadatas[0] || [];
  const dists = results.distances[0] || [];
  const embs = results.embeddings[0] || [];
  
  if (docs.length === 0) {
    console.timeEnd('[PERF] Total retrieval');
    return [];
  }
  
  // MMR pour diversité
  console.time('[PERF] MMR');
  const mmrCount = Math.min(kFinal * 2, docs.length);
  const mmrIdx = mmrOptimized(queryEmbedding, embs, mmrCount, 0.6);
  console.timeEnd('[PERF] MMR');
  
  // Préparer le subset pour reranking
  const subset = mmrIdx.map(i => ({
    text: docs[i],
    source: metas[i]?.source || 'inconnu',
    rawDistance: dists[i],
    cosSim: vectorOps.cosine(queryEmbedding, embs[i])
  }));
  
  // Reranking - avec fallback rapide
  let reranked;
  
  // Skip reranking si peu de résultats
  if (subset.length <= kFinal) {
    console.log('[OPTIM] Skip reranking - peu de résultats');
    reranked = subset.map(c => ({ ...c, rerank: c.cosSim }));
  } else {
    console.time('[PERF] Reranking');
    try {
      // Timeout pour le reranking
      const rerankerPromise = rerankCrossEncoder(query, subset);
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Timeout')), 3000)
      );
      
      reranked = await Promise.race([rerankerPromise, timeoutPromise]);
      console.timeEnd('[PERF] Reranking');
    } catch (e) {
      console.warn('[RERANK] Fallback lexical:', e.message);
      console.time('[PERF] Lexical fallback');
      
      // Fallback hybride rapide
      reranked = subset.map(c => ({
        ...c,
        rerank: 0.7 * c.cosSim + 0.3 * lexicalScoreFast(query, c.text)
      })).sort((a, b) => b.rerank - a.rerank);
      
      console.timeEnd('[PERF] Lexical fallback');
    }
  }
  
  console.timeEnd('[PERF] Total retrieval');
  
  // Retourner top-k
  return reranked.slice(0, kFinal).map(c => ({
    text: c.text,
    source: c.source,
    rawDistance: c.rawDistance,
    score: c.rerank || c.cosSim
  }));
}

// Fonction de préchauffage au démarrage
async function warmup() {
  console.log('[WARMUP] Préchauffage du système...');
  try {
    await initCollection();
    // Faire une requête bidon pour charger les modèles
    await retrieveRelevant('test', 1, 3);
    console.log('[WARMUP] Système prêt!');
  } catch (e) {
    console.warn('[WARMUP] Échec du préchauffage:', e.message);
  }
}

// Lancer le warmup au démarrage
setTimeout(warmup, 1000);

module.exports = { addDocument, retrieveRelevant };