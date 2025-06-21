const express = require('express');
const cors = require('cors');
const { retrieveRelevant } = require('./rag/retriever');

// Utilisation du fetch natif de Node.js (version 18+)
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));

const app = express();
const PORT = 3001;

// Configuration CORS simplifiée
app.use(cors({
  origin: '*', // Temporairement permissif pour les tests
  methods: ['POST', 'GET', 'OPTIONS'],
  allowedHeaders: ['Content-Type']
}));

app.use(express.json());

// Middleware de sécurité
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  next();
});

// Route de santé
app.get('/health', (req, res) => {
  res.status(200).send('OK');
});

// Route principale
app.post('/ask', async (req, res) => {
  console.log('[LOG] Requête reçue sur /ask');
  
  try {
    const rawPrompt = req.body.prompt?.trim() || '';
    if (!rawPrompt) return res.status(400).send('Prompt requis');

    console.log('Question:', rawPrompt);

    const relevantChunks = await retrieveRelevant(rawPrompt);
    const context = relevantChunks.map(chunk => `• ${chunk.text}`).join('\n');
    const finalPrompt = `Réponds en première personne (tu es Léo)\nContexte:\n${context}\n\nQuestion: ${rawPrompt}`;

    console.log('Prompt envoyé à Ollama:', finalPrompt.substring(0, 200) + (finalPrompt.length > 200 ? '...' : ''));

    // Appel à Ollama
    const ollamaResponse = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'openchat',
        prompt: finalPrompt,
        stream: true
      })
    });

    // Gestion des erreurs Ollama
    if (!ollamaResponse.ok) {
      const errorText = await ollamaResponse.text();
      throw new Error(`Ollama error: ${ollamaResponse.status} - ${errorText}`);
    }

    // Configuration de la réponse
    res.setHeader('Content-Type', 'text/plain');
    res.setHeader('Cache-Control', 'no-cache');

    // Streaming de la réponse
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
    console.log('✅ Réponse envoyée');

  } catch (err) {
    console.error('ERREUR:', err);
    if (!res.headersSent) {
      res.status(500).send('Erreur serveur: ' + err.message);
    }
  }
});

// Démarrer le serveur
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Serveur démarré sur http://localhost:${PORT}`);
});