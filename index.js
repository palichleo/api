// index.js

const express = require('express');
const cors = require('cors');
const { retrieveRelevant } = require('./rag/retriever');

const app = express();
const PORT = 3000;

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

app.get('/health', (req, res) => {
  res.status(200).send('OK');
});

app.post('/ask', async (req, res) => {
  console.log('[LOG] Requête reçue sur /ask');

  try {
    const rawPrompt = req.body.prompt?.trim() || '';
    if (!rawPrompt) return res.status(400).send('Prompt requis');
    console.log('Question:', rawPrompt);

    // On laisse retriever faire kInitial=12 → MMR → rerank → Top‑3
    const relevant = await retrieveRelevant(rawPrompt, 3, 12);
    console.log('Chunks trouvés:', relevant.map((c, i) => ({ i, source: c.source, score: c.score.toFixed(3) })));

    const bullets = relevant.map((c, i) => `• [${i+1}] (source: ${c.source})\n${c.text}`).join('\n\n');

    const finalPrompt =
`Tu es Léo Palich. Réponds UNIQUEMENT en te basant sur les informations fournies ci-dessous.

RÈGLES IMPORTANTES :
- Tu es Léo Palich, étudiant en Sciences cognitives IA Centrée Humain
- Utilise EXCLUSIVEMENT les informations des extraits fournis
- Réponds en première personne ("Je suis...", "Mon numéro est...", etc.)
- Si l'information n'est pas dans les extraits, dis "Cette information n'est pas disponible dans mes données"
- Sois concis et direct

[EXTRAITS]
${bullets}

[QUESTION] ${rawPrompt}
`;

    console.log('Prompt envoyé à Ollama:', finalPrompt.substring(0, 200) + (finalPrompt.length > 200 ? '...' : ''));

    const ollamaResponse = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'phi3:mini',
        prompt: finalPrompt,
        stream: true,
        options: {
          temperature: 0.1,
          top_p: 0.8,
          repeat_penalty: 1.1,
          num_ctx: 2048,
          num_predict: 200,
          num_thread: 0
        }
      })
    });

    if (!ollamaResponse.ok) {
      const errorText = await ollamaResponse.text();
      throw new Error(`Ollama error: ${ollamaResponse.status} - ${errorText}`);
    }

    res.setHeader('Content-Type', 'text/plain');
    res.setHeader('Cache-Control', 'no-cache');

    const reader = ollamaResponse.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(line => line.trim());

      for (const line of lines) {
        try {
          const data = JSON.parse(line);
          if (data.response) {
            res.write(data.response);
          }
        } catch (e) {
          console.error('Erreur parsing JSON:', e.message, 'Ligne:', line);
        }
      }
    }

    res.end();
    console.log('Réponse envoyée');

  } catch (err) {
    console.error('ERREUR:', err);
    if (!res.headersSent) {
      res.status(500).send('Erreur serveur: ' + err.message);
    }
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Serveur démarré sur http://localhost:${PORT}`);
});
