// rag/indexer.js

const fs = require('fs');
const path = require('path');
const { addDocument } = require('./retriever');

const KNOWLEDGE_DIR = path.join(__dirname, '../knowledges');

// Découpage en sections à partir des titres markdown
function splitMarkdownBySections(text) {
  const sections = text.split(/^#{1,3} /gm)
    .map(s => s.trim())
    .filter(s => s.length > 30); // Ignore les mini-sections

  return sections.map(s => '# ' + s); // Réinjecte le titre
}

// Fonction principale d’indexation
async function indexKnowledge() {
  const files = fs.readdirSync(KNOWLEDGE_DIR).filter(f =>
    f.endsWith('.md') || f.endsWith('.txt')
  );

  for (const file of files) {
    const content = fs.readFileSync(path.join(KNOWLEDGE_DIR, file), 'utf8');
    const chunks = splitMarkdownBySections(content);

    for (const chunk of chunks) {
      console.log(`📄 [${file}] →`, chunk.slice(0, 60).replace(/\n/g, ' ') + '...');
      await addDocument(chunk, file);
    }
  }

  console.log(`${files.length} fichier(s) indexé(s) depuis knowledges/`);
}

indexKnowledge().catch(err => {
  console.error('Erreur lors de l’indexation :', err);
});