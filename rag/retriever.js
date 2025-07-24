// rag/retriever.js
const { ChromaClient, LocalEmbeddingFunction } = require('chromadb');
const { embed } = require('./embedder');

const chroma = new ChromaClient({ host: 'localhost', port: 8000 });

const embeddingFunction = new LocalEmbeddingFunction(async (texts) => {
  const embeddings = [];
  for (const text of texts) {
    embeddings.push(await embed(text));
  }
  return embeddings;
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
  const embedding = await embed(text);

  await col.add({
    ids: [Date.now().toString()],
    documents: [text],
    metadatas: [{ source }],
    embeddings: [embedding],
  });
}

async function retrieveRelevant(query, k = 3) {
  const col = await initCollection();
  const queryEmbedding = await embed(query);

  const results = await col.query({
    queryEmbeddings: [queryEmbedding],
    nResults: k,
  });

  return results.documents[0].map((text, i) => ({
    text,
    source: results.metadatas[0][i]?.source || 'inconnu',
    score: results.distances[0][i],
  }));
}

module.exports = { addDocument, retrieveRelevant };