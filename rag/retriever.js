const { ChromaClient } = require('chromadb');
const { embed } = require('./embedder');

const chroma = new ChromaClient({ path: 'http://localhost:8000' });
let collection;

async function initCollection() {
  if (!collection) {
    await chroma.createCollection({ name: 'leoknowledge' }).catch(() => {});
    collection = await chroma.getCollection({ name: 'leoknowledge' });
  }
  return collection;
}

async function addDocument(text, source = 'unknown') {
  const embedding = await embed(text);
  const coll = await initCollection();
  await coll.add({
    ids: [Date.now().toString() + Math.random()],
    documents: [text],
    metadatas: [{ source }],
    embeddings: [embedding]
  });
}

async function retrieveRelevant(query, k = 3) {
  const coll = await initCollection();
  const queryVec = await embed(query);
  const results = await coll.query({
    queryEmbeddings: [queryVec],
    nResults: k
  });

  return results.documents[0].map((text, i) => ({
    text,
    source: results.metadatas[0][i]?.source,
    score: results.distances[0][i]
  }));
}

module.exports = { addDocument, retrieveRelevant };