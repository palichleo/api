// index.js
const express = require('express');
const cors = require('cors');
const os = require('os');
const { retrieveRelevant } = require('./rag/retriever');
require('dotenv').config();

const app = express();
const PORT = 3000;

const hr = () => Number(process.hrtime.bigint()) / 1e6; // ms

// Ajoute des timings soit en header (avant envoi), soit en trailer (après envoi)
function addServerTiming(res, metrics, { trailer = false } = {}) {
  const parts = Object.entries(metrics).map(([k, v]) => `${k};dur=${v}`);
  const value = parts.join(', ');
  try {
    if (trailer) {
      if (typeof res.addTrailers === 'function') {
        res.addTrailers({ 'Server-Timing': value });
      }
      return;
    }
    if (res.headersSent) {
      if (typeof res.addTrailers === 'function') {
        res.addTrailers({ 'Server-Timing': value });
      }
      return;
    }
    const prev = res.getHeader('Server-Timing');
    res.setHeader('Server-Timing', prev ? String(prev) + ', ' + value : value);
  } catch (_) {}
}

function trunc(s, max = 800) {
  if (!s || s.length <= max) return s || '';
  return s.slice(0, max) + '…';
}

async function warmup() {
  try {
    const model = process.env.OLLAMA_MODEL || 'phi3:mini';
    await fetch('http://127.0.0.1:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        prompt: 'Bonjour',
        stream: true,
        keep_alive: '1h',
        options: {
          temperature: 0.1,
          top_p: 0.9,
          repeat_penalty: 1.1,
          num_ctx: 1024,
          num_predict: 60,
          num_thread: 4,
          num_batch: 16
        }
      })
    });
    console.log(`[WARMUP] OK (${model})`);
  } catch (e) {
    console.warn('[WARMUP] skip:', e.message);
  }
}

app.use(cors({
  origin: '*',
  methods: ['POST', 'GET', 'OPTIONS'],
  allowedHeaders: ['Content-Type']
}));
app.use(express.json());
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  next();
});

app.get('/health', (req, res) => res.status(200).send('OK'));

app.post('/ask', async (req, res) => {
  console.log('[LOG] Requête reçue sur /ask');
  try {
    const rawPrompt = (req.body?.prompt || '').toString().trim();
    if (!rawPrompt) return res.status(400).send('Prompt requis');
    console.log('Question:', rawPrompt);

    const dbg = {};
    const t0 = hr();
    const relevant = await retrieveRelevant(rawPrompt, 3, 12, dbg);
    const t1 = hr();
    dbg.t_retrieve_ms = +(t1 - t0).toFixed(2);

    console.log('Chunks trouvés:', relevant.map((c, i) => ({ i, source: c.source, score: c.score?.toFixed?.(3) || c.score })));

    const context = relevant.map((c, idx) =>
      `- [${idx + 1}] (${c.source}) ${trunc(c.text.replace(/\s+/g, ' ').trim(), 800)}`
    ).join('\n');

    const finalPrompt =
`Tu es Léo Palich. Réponds UNIQUEMENT en te basant sur les informations fournies ci-dessous.

RÈGLES IMPORTANTES :
- Tu es Léo Palich, étudiant en Sciences cognitives IA Centrée Humain
- Utilise EXCLUSIVEMENT les informations des extraits fournis
- Réponds en première personne ("Je suis...", "Mon numéro est...", etc.)
- Si l'information n'est pas dans les extraits, dis "Cette information n'est pas disponible dans mes données"
- Sois concis et direct

[EXTRAITS]
${context}

[QUESTION] ${rawPrompt}
`;

    console.log('Prompt envoyé à Ollama:', finalPrompt.substring(0, 200) + (finalPrompt.length > 200 ? '...' : ''));

    const t2 = hr();
    const model = process.env.OLLAMA_MODEL || 'phi3:mini';
    const ollamaRes = await fetch('http://127.0.0.1:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        prompt: finalPrompt,
        stream: true,
        keep_alive: '1h', // ← top-level (PAS dans options)
        options: {
          temperature: 0.1,
          top_p: 0.9,
          repeat_penalty: 1.1,
          num_ctx: 1024,
          num_predict: 80,
          num_thread: 4,
          num_batch: 16
        }
      })
    });
    const t3 = hr();
    dbg.t_ollama_req_ms = +(t3 - t2).toFixed(2);

    if (!ollamaRes.ok || !ollamaRes.body) {
      const errorText = await ollamaRes.text().catch(() => '');
      throw new Error(`Ollama error: ${ollamaRes.status} - ${errorText}`);
    }

    // Headers streaming (déclarés une seule fois avant tout res.write)
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Transfer-Encoding', 'chunked');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');   // Nginx: désactive le buffering
    res.setHeader('Trailer', 'Server-Timing');  // timings envoyés en fin de flux

    // Timings "pré-stream" (en header, avant écriture du corps)
    addServerTiming(res, {
      emb: dbg.t_embed_ms ?? 0,
      chroma: dbg.t_chroma_ms ?? 0,
      mmr: dbg.t_mmr_ms ?? 0,
      retrieve: dbg.t_retrieve_ms ?? 0,
      prep: +(t2 - t1).toFixed(2),
      req: dbg.t_ollama_req_ms ?? 0
    });

    let ttft_ms = null;
    let lastChunk = null;
    const t_stream_start = hr();
    const reader = ollamaRes.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    // Timeout d’inactivité sur le stream (ex: 20 s sans token -> fin)
    const STREAM_IDLE_MS = 20000;
    let idleTimer = setTimeout(() => {
      try { res.write('\n\n[stream idle timeout]\n'); } catch (_) {}
      try { res.end(); } catch (_) {}
      console.error('[STREAM] Idle timeout, fin de réponse.');
    }, STREAM_IDLE_MS);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        try { res.write('\n\n[stream idle timeout]\n'); } catch (_) {}
        try { res.end(); } catch (_) {}
        console.error('[STREAM] Idle timeout, fin de réponse.');
      }, STREAM_IDLE_MS);

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const data = JSON.parse(line);
          if (ttft_ms === null) ttft_ms = +(hr() - t_stream_start).toFixed(2);
          lastChunk = data;
          if (data.response) res.write(data.response);
        } catch (e) {
          console.warn('[STREAM] ligne non JSON:', line.slice(0, 200));
        }
      }
    }

    clearTimeout(idleTimer);

    const t_done = hr();
    const gen_ms = +(t_done - t_stream_start).toFixed(2);
    let tok_s = null;
    if (lastChunk && typeof lastChunk.eval_count === 'number' && typeof lastChunk.eval_duration === 'number') {
      const evalSec = lastChunk.eval_duration / 1e9;
      tok_s = evalSec > 0 ? +(lastChunk.eval_count / evalSec).toFixed(2) : null;
    }

    // Timings "post-stream" : envoyés en TRAILER
    addServerTiming(res, {
      ttft: ttft_ms ?? 0,
      gen: gen_ms,
      toks: tok_s ?? 0
    }, { trailer: true });

    res.end();
    console.log('[TIMINGS]', {
      t_embed_ms: dbg.t_embed_ms,
      t_chroma_ms: dbg.t_chroma_ms,
      t_mmr_ms: dbg.t_mmr_ms,
      t_retrieve_ms: dbg.t_retrieve_ms,
      t_prep_ms: +(t2 - t1).toFixed(2),
      t_ollama_req_ms: dbg.t_ollama_req_ms,
      t_ttft_ms: ttft_ms,
      t_gen_ms: gen_ms,
      tokens_per_s: tok_s
    });
  } catch (err) {
    console.error('ERREUR:', err);
    if (!res.headersSent) res.status(500).send('Erreur serveur: ' + err.message);
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Serveur démarré sur http://localhost:${PORT}`);
  warmup();
});
