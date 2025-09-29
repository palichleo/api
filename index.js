// index.js
const express = require('express');
const cors = require('cors');
const { retrieveRelevant } = require('./rag/retriever');
require('dotenv').config({ path: require('path').join(__dirname, '.env') });

const GROQ_MODEL = process.env.GROQ_MODEL || 'openai/gpt-oss-120b';
const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';

const app = express();
const PORT = process.env.PORT || 3000;

function sanitizeOut(s = '') {
  return s
    // vire les styles Markdown
    .replace(/\*\*/g, '')
    .replace(/\*/g, '')
    .replace(/__+/g, '')     // italiques/gras avec underscores
    .replace(/`+/g, '')      // code inline
    // guillemets droits + espaces propres
    .replace(/[“”]/g, '"').replace(/[‘’]/g, "'")
    .replace(/\s{2,}/g, ' ');
}

function trunc(s, max = 800) {
  if (!s || s.length <= max) return s || '';
  return s.slice(0, max) + '…';
}

app.use(cors({ origin: '*', methods: ['POST', 'GET', 'OPTIONS'], allowedHeaders: ['Content-Type'] }));
app.use(express.json());
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  next();
});

app.get('/health', (req, res) => res.status(200).send('OK'));

async function streamGroq(res, systemPrompt, userPrompt) {
  const r = await fetch(GROQ_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.GROQ_API_KEY}`
    },
    body: JSON.stringify({
      model: GROQ_MODEL,
      stream: true,
      temperature: 0.0,
      top_p: 1,
      max_tokens: 900,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ]
    })
  });
  if (!r.ok || !r.body) {
    const errorText = await r.text().catch(() => '');
    throw new Error(`Groq error: ${r.status} - ${errorText}`);
  }

  // headers streaming
  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.setHeader('Transfer-Encoding', 'chunked');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');

  const reader = r.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const l = line.trim();
      if (!l.startsWith('data:')) continue;
      const data = l.slice(5).trim();
      if (!data || data === '[DONE]') continue;
      try {
        const evt = JSON.parse(data);
        const tok = evt.choices?.[0]?.delta?.content;
        if (tok) res.write(sanitizeOut(tok));
      } catch (_) {}
    }
  }
  res.end();
}

app.post('/ask', async (req, res) => {
  try {
    if (!process.env.GROQ_API_KEY) return res.status(500).send('GROQ_API_KEY manquant');
    const rawPrompt = (req.body?.prompt || '').toString().trim();
    if (!rawPrompt) return res.status(400).send('Prompt requis');

    const relevant = await retrieveRelevant(rawPrompt, 10, 63, {});
    const context = relevant.slice(0, 10).map((c, idx) =>
      `- [${idx + 1}] (${c.source}) ${trunc(c.text.replace(/\s+/g, ' ').trim(), 1600)}`
    ).join('\n');

    const today = new Date().toISOString().slice(0, 10);
    const systemPrompt =
      `Tu es un assistant RAG francophone ultra-strict.\n` +
      `RÈGLES:\n` +
      `- Tu réponds UNIQUEMENT à partir des EXTRACTS fournis.\n`+
      `- Si les EXTRACTS sont insuffisants, réponds exactement: "Hors corpus: information manquante".\n`+
      `- Ne fais AUCUNE supposition. Pas d'invention de dates, chiffres, références.\n`+
      `- Cite les passages utilisés avec [n] où n correspond à la liste dans EXTRACTS.\n`+
      `- Réponds en français, clair et factuel (≤180 mots). Aujourd'hui: ${today}.`;

  const userPrompt =
    `EXTRACTS (numérotés):\n${context}\n\n` +
    `QUESTION:\n${rawPrompt}\n\n` +
    `INSTRUCTIONS:\n` +
    `1) Ne répondre que si l'information apparaît explicitement dans au moins un EXTRACT.\n` +
    `2) Dans la réponse, insérer les citations [n] au bon endroit.\n` +
    `3) Si manque: "Hors corpus: information manquante".`;
    
    await streamGroq(res, systemPrompt, userPrompt);
  } catch (err) {
    console.error('ERREUR:', err);
    if (!res.headersSent) res.status(500).send('Erreur serveur: ' + err.message);
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Serveur Groq prêt sur http://localhost:${PORT} (modèle: ${GROQ_MODEL})`);
});
