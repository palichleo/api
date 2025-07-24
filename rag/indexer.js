// rag/indexer.js
const fs = require('fs');
const path = require('path');
const { addDocument } = require('./retriever');
const vectorPath = path.join(__dirname, 'store.json');
const KNOWLEDGE_DIR = path.join(__dirname, '../knowledges');
const { ChromaClient } = require('chromadb');

const chroma = new ChromaClient({ path: 'http://localhost:8000' });

await chroma.deleteCollection({ name: 'leoknowledge' }).catch(() => {});
await chroma.createCollection({ name: 'leoknowledge' });

function splitMarkdownBySections(text) {
  const sections = text.split(/^#{1,3} /gm) // découpe par #, ##, ### 
    .map(s => s.trim())
    .filter(s => s.length > 30); // filtre les morceaux trop petits ou vides

  return sections.map(s => '# ' + s); // remet une balise titre pour chaque chunk
}

async function indexKnowledge() {
  fs.writeFileSync(vectorPath, '[]'); // reset store

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

  console.log(`${files.length} fichiers indexés depuis le dossier knowledges/`);
}

indexKnowledge();