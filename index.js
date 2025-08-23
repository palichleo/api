// index.js - Version optimisée
const express = require('express');
const cors = require('cors');
const { retrieveRelevant } = require('./rag/retriever');

const app = express();
const PORT = 3000;

// Cache LRU simple pour les réponses fréquentes
const responseCache = new Map();
const CACHE_SIZE = 50;
const CACHE_TTL = 3600000; // 1 heure

app.use(cors({
  origin: '*',
  methods: ['POST', 'GET', 'OPTIONS'],
  allowedHeaders: ['Content-Type']
}));
app.use(express.json());

app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  // Fix encodage UTF-8
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  next();
});

app.get('/health', (req, res) => {
  res.status(200).send('OK');
});

// Normalisation des questions pour le cache
function normalizeQuery(text) {
  return text.toLowerCase().trim().replace(/\s+/g, ' ');
}

app.post('/ask', async (req, res) => {
  console.log('[LOG] Requête reçue sur /ask');

  try {
    const rawPrompt = req.body.prompt?.trim() || '';
    if (!rawPrompt) return res.status(400).send('Prompt requis');
    console.log('Question:', rawPrompt);

    // Vérifier le cache
    const cacheKey = normalizeQuery(rawPrompt);
    const cached = responseCache.get(cacheKey);
    if (cached && (Date.now() - cached.timestamp < CACHE_TTL)) {
      console.log('[CACHE HIT] Réponse depuis le cache');
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.setHeader('X-Cache', 'HIT');
      return res.send(cached.response);
    }

    // Paramètres optimisés : moins de chunks pour CPU
    const relevant = await retrieveRelevant(rawPrompt, 2, 8); // Réduit de 3,12 à 2,8
    console.log('Chunks trouvés:', relevant.map((c, i) => ({ 
      i, 
      source: c.source, 
      score: c.score.toFixed(3) 
    })));

    const bullets = relevant.map((c, i) => 
      `• [${i+1}] (source: ${c.source})\n${c.text}`
    ).join('\n\n');

    const finalPrompt = `Tu es Léo Palich. Réponds UNIQUEMENT en te basant sur les informations fournies ci-dessous.

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

    console.log('Prompt envoyé à Ollama:', finalPrompt.substring(0, 200) + '...');

    // Optimisations Ollama pour CPU
    const ollamaResponse = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gemma2:2b', // Modèle léger pour CPU
        prompt: finalPrompt,
        stream: true,
        options: {
          temperature: 0.1,
          top_p: 0.8,
          repeat_penalty: 1.1,
          num_ctx: 1024,      // Réduit de 2048
          num_predict: 150,   // Réduit de 200
          num_thread: 0,      // Laisse Ollama gérer
          num_gpu: 0,         // Force CPU
          f16_kv: false,      // Désactive FP16 pour CPU
          use_mmap: true,     // Active memory mapping
          use_mlock: false    // Évite le verrouillage mémoire
        }
      })
    });

    if (!ollamaResponse.ok) {
      const errorText = await ollamaResponse.text();
      throw new Error(`Ollama error: ${ollamaResponse.status} - ${errorText}`);
    }

    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('X-Cache', 'MISS');

    const reader = ollamaResponse.body.getReader();
    const decoder = new TextDecoder('utf-8'); // Force UTF-8
    let fullResponse = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n').filter(line => line.trim());

      for (const line of lines) {
        try {
          const data = JSON.parse(line);
          if (data.response) {
            res.write(data.response);
            fullResponse += data.response;
          }
        } catch (e) {
          console.error('Erreur parsing JSON:', e.message);
        }
      }
    }

    // Stocker dans le cache
    if (fullResponse) {
      responseCache.set(cacheKey, {
        response: fullResponse,
        timestamp: Date.now()
      });
      
      // Limiter la taille du cache
      if (responseCache.size > CACHE_SIZE) {
        const firstKey = responseCache.keys().next().value;
        responseCache.delete(firstKey);
      }
    }

    res.end();
    console.log('Réponse envoyée');

  } catch (err) {
    console.error('ERREUR:', err);
    if (!res.headersSent) {
      res.status(500).json({ error: 'Erreur serveur', message: err.message });
    }
  }
});

// Endpoint pour vider le cache
app.post('/cache/clear', (req, res) => {
  responseCache.clear();
  res.json({ message: 'Cache vidé', size: 0 });
});

// Endpoint pour les stats du cache
app.get('/cache/stats', (req, res) => {
  res.json({ 
    size: responseCache.size, 
    maxSize: CACHE_SIZE,
    ttl: CACHE_TTL 
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Serveur démarré sur http://localhost:${PORT}`);
  console.log(`Cache activé: ${CACHE_SIZE} entrées max, TTL: ${CACHE_TTL/1000}s`);
});