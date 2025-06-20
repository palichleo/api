const fs = require('fs');
const path = require('path');
const { embed } = require('./embedder');

const storePath = path.resolve(__dirname, 'store.json');

function cosineSimilarity(vecA, vecB) {
  const dot = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dot / (normA * normB);
}

function loadStore() {
  return JSON.parse(fs.readFileSync(storePath, 'utf8'));
}

function saveStore(data) {
  fs.writeFileSync(storePath, JSON.stringify(data, null, 2));
}

async function addDocument(text, source = 'unknown') {
  const embedding = await embed(text);
  const store = loadStore();
  store.push({ text, embedding, source });
  saveStore(store);
}

async function retrieveRelevant(query, k = 3) {
  const store = loadStore();
  const queryVec = await embed(query);
  return store
    .map(item => ({ ...item, score: cosineSimilarity(queryVec, item.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

module.exports = { addDocument, retrieveRelevant };