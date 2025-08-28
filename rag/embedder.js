// rag/embedder.js
const fetch = require('node-fetch');

const PROVIDER = (process.env.EMBED_PROVIDER || 'openai').toLowerCase();

// --- OpenAI
async function openaiEmbedMany(texts) {
  const model = process.env.OPENAI_EMBED_MODEL || 'text-embedding-3-large'; // qualité
  const r = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`
    },
    body: JSON.stringify({ model, input: texts })
  });
  if (!r.ok) {
    const t = await r.text().catch(()=> '');
    throw new Error(`OpenAI embed error: ${r.status} - ${t}`);
  }
  const data = await r.json();
  return data.data.map(d => d.embedding);
}

// --- Mistral
async function mistralEmbedMany(texts) {
  const model = process.env.MISTRAL_EMBED_MODEL || 'mistral-embed';
  const r = await fetch('https://api.mistral.ai/v1/embeddings', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.MISTRAL_API_KEY}`
    },
    body: JSON.stringify({ model, input: texts })
  });
  if (!r.ok) {
    const t = await r.text().catch(()=> '');
    throw new Error(`Mistral embed error: ${r.status} - ${t}`);
  }
  const data = await r.json();
  // { data: [{embedding: [...]}...] }
  return data.data.map(d => d.embedding);
}

async function embedMany(texts) {
  if (!Array.isArray(texts)) texts = [texts];
  if (PROVIDER === 'mistral') return await mistralEmbedMany(texts);
  // default openai
  return await openaiEmbedMany(texts);
}

async function embed(text) {
  const arr = await embedMany([text]);
  return arr[0];
}

module.exports = { embed, embedMany };
