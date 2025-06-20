const express = require('express');
const cors = require('cors');
const fs = require('fs');
const { retrieveRelevant } = require('./rag/retriever');

const app = express();
const PORT = 3001;

app.use(cors({
  origin: 'https://www.leopalich.com'
}));
app.use(express.json());

app.post('/ask', async (req, res) => {
  const rawPrompt = req.body.prompt?.trim() || '';
  console.log('\nQuestion utilisateur :', rawPrompt);

  try {
    const relevantChunks = await retrieveRelevant(rawPrompt);
    const context = relevantChunks.map(chunk => `• ${chunk.text}`).join('\n');

    const finalPrompt = `Réponds uniquement par la premiere personne de l'indicatif (tu es Léo)\nConnaissances utiles :\n${context}\n\nQuestion : ${rawPrompt}`;

    console.log('\nPrompt envoyé à Ollama : ');
    console.log(finalPrompt);
    console.log('\nFin du prompt\n');

    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'openchat',
        prompt: finalPrompt,
        stream: true
      })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullResponse = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });

      // chaque ligne est un JSON du style : { "response": "bla", "done": false }
      chunk.split('\n').forEach(line => {
        try {
          const json = JSON.parse(line);
          if (json.response) {
            fullResponse += json.response;
          }
        } catch (e) {}
      });
    }

    res.json({ response: fullResponse });

    const result = await response.json();
    res.json({ response: result.response });
  } catch (err) {
    console.error('Erreur API ou RAG :', err);
    res.status(500).json({ error: 'Erreur serveur interne' });
  }
});

app.listen(PORT, () => {
  console.log(`API LéoBot avec RAG active sur http://localhost:${PORT}`);
});