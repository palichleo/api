const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch'); // Important : ajout de node-fetch
const { retrieveRelevant } = require('./rag/retriever');

const app = express();
const PORT = 3001;

// Configuration améliorée du CORS
app.use(cors({
  origin: [
    'https://www.leopalich.com',
    'https://api.leopalich.com'
  ],
  methods: ['POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type']
}));

// Middleware pour les pré-requêtes OPTIONS
app.options('*', cors());

app.use(express.json());

// Middleware de sécurité supplémentaire
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  next();
});

app.post('/ask', async (req, res) => {
  console.log('[LOG] Requête reçue sur /ask');
  
  try {
    const rawPrompt = req.body.prompt?.trim() || '';
    if (!rawPrompt) {
      return res.status(400).json({ error: 'Le prompt est requis' });
    }

    console.log('\nQuestion utilisateur :', rawPrompt);

    const relevantChunks = await retrieveRelevant(rawPrompt);
    const context = relevantChunks.map(chunk => `• ${chunk.text}`).join('\n');
    const finalPrompt = `Réponds uniquement par la premiere personne de l'indicatif (tu es Léo)\nConnaissances utiles :\n${context}\n\nQuestion : ${rawPrompt}`;

    console.log('\nPrompt envoyé à Ollama :', finalPrompt.substring(0, 200) + '...');

    // Appel à Ollama avec timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // Timeout après 30s

    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'openchat',
        prompt: finalPrompt,
        stream: true
      }),
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Ollama error: ${response.status} ${response.statusText}`);
    }

    // Configuration des headers de flux
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache');
    res.flushHeaders();

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n').filter(line => line.trim());

      for (const line of lines) {
        try {
          const { response: ollamaResponse } = JSON.parse(line);
          if (ollamaResponse) {
            res.write(ollamaResponse);
          }
        } catch (err) {
          console.error('Erreur parsing JSON:', line);
        }
      }
    }

    res.end();
    console.log('✅ Réponse envoyée');

  } catch (err) {
    console.error('ERREUR CRITIQUE:', err);
    
    if (err.name === 'AbortError') {
      if (!res.headersSent) res.status(504).send('Timeout du serveur Ollama');
    } else {
      if (!res.headersSent) res.status(500).send('Erreur serveur');
    }
    
    if (!res.headersSent) {
      res.end();
    }
  }
});

// Route de santé pour les tests
app.get('/health', (req, res) => {
  res.status(200).send('OK');
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Serveur démarré sur le port ${PORT}`);
});