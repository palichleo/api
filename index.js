// index.js
const express = require('express');
const cors = require('cors');
const { retrieveRelevant } = require('./rag/retriever');
require('dotenv').config();

const GROQ_MODEL = process.env.GROQ_MODEL || 'llama-3.1-8b-instant';
const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';

const app = express();
const PORT = process.env.PORT || 3000;

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
      temperature: 0.1,
      max_tokens: 256,
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
        if (tok) res.write(tok);
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

    const relevant = await retrieveRelevant(rawPrompt, 3, 12, {});
    const context = relevant.slice(0, 1).map((c, idx) =>
      `- [${idx + 1}] (${c.source}) ${trunc(c.text.replace(/\s+/g, ' ').trim(), 250)}`
    ).join('\n');

    const today = new Date().toISOString().slice(0, 10);
    const systemPrompt =
      `Tu es Léo Palich. Réponds UNIQUEMENT à partir des extraits. ` +
      `Sois concis (≤40 mots). NOW: ${today}. Respecte la temporalité (passé si date < NOW).`;

    const userPrompt = `[EXTRAITS]\n${context}\n\n[QUESTION] ${rawPrompt}`;

    await streamGroq(res, systemPrompt, userPrompt);
  } catch (err) {
    console.error('ERREUR:', err);
    if (!res.headersSent) res.status(500).send('Erreur serveur: ' + err.message);
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Serveur Groq prêt sur http://localhost:${PORT} (modèle: ${GROQ_MODEL})`);
});
