// rag/retriever.js
const { ChromaClient, LocalEmbeddingFunction } = require('chromadb');
const { embed } = require('./embedder');

const chroma = new ChromaClient({ host: 'localhost', port: 8000 });

// Définition de notre fonction d'embedding personnalisée
const embeddingFunction = new LocalEmbeddingFunction(async (texts) => {
  if (typeof texts === 'string') {
    return [await embed(texts)];
  } else {
    const vectors = [];
    for (const t of texts) {
      vectors.push(await embed(t));
    }
    return vectors;
  }
});

let collection;

async function initCollection() {
  if (!collection) {
    try {
      collection = await chroma.getCollection({ name: 'leoknowledge' });
    } catch {
      collection = await chroma.createCollection({
        name: 'leoknowledge',
        embeddingFunction,
      });
    }
  }
  return collection;
}

async function addDocument(text, source = 'unknown') {
  const col = await initCollection();
  await col.add({
    documents: [text],
    ids: [Date.now().toString()],
    metadatas: [{ source }],
  });
}

async function retrieveRelevant(query, k = 3) {
  const col = await initCollection();
  const results = await col.query({
    queryTexts: [query],
    nResults: k,
  });

  return results.documents[0].map((text, i) => ({
    text,
    source: results.metadatas[0][i]?.source || 'inconnu',
    score: results.distances[0][i],
  }));
}

module.exports = { addDocument, retrieveRelevant };