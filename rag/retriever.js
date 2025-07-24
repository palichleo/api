// rag/retriever.js
const { ChromaClient } = require('chromadb');
const { embed } = require('./embedder');

const chroma = new ChromaClient({ host: 'localhost', port: 8000 });
let collection;

async function initCollection() {
  if (!collection) {
    collection = await chroma.getOrCreateCollection({ name: 'leoknowledge' });
  }
  return collection;
}

async function addDocument(text, source = 'unknown') {
  const embedding = await embed(text);
  const coll = await initCollection();
  await coll.add({
    ids: [crypto.randomUUID()],
    embeddings: [embedding],
    documents: [text],
    metadatas: [{ source }]
  });
}

async function retrieveRelevant(query, k = 3) {
  const queryVec = await embed(query);
  const coll = await initCollection();
  const results = await coll.query({
    query_embeddings: [queryVec],
    n_results: k
  });
  return results.documents[0].map((text, i) => ({
    text,
    score: results.distances[0][i],
    source: results.metadatas[0][i]?.source || 'unknown'
  }));
}

module.exports = { addDocument, retrieveRelevant };