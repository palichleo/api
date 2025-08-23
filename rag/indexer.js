// rag/indexer.js
const fs = require('fs');
const path = require('path');
const { addDocuments } = require('./retriever');

const ROOT = path.resolve(__dirname, '..', 'knowledges');
const BATCH_SIZE = parseInt(process.env.INDEX_BATCH || '64', 10);

function splitMarkdownSmart(text, {
  targetTokensMin = 50,
  targetTokensMax = 280,
  overlapTokens = 20
} = {}) {
  // version simple et robuste (garde les titres, évite de couper les fences)
  const lines = text.split('\n');
  const chunks = [];
  let buf = [];
  let tokens = 0;
  let inFence = false;

  const flush = () => {
    if (buf.length === 0) return;
    const chunk = buf.join('\n').trim();
    if (chunk) chunks.push(chunk);
    buf = [];
    tokens = 0;
  };

  for (const line of lines) {
    if (line.trim().startsWith('```')) inFence = !inFence;
    const add = line + '\n';
    const addTokens = Math.max(1, Math.ceil(add.split(/\s+/).length * 0.9));
    if (!inFence && tokens + addTokens > targetTokensMax) {
      flush();
      // overlap simple
      if (overlapTokens > 0) {
        const words = line.split(/\s+/);
        buf = words.slice(Math.max(0, words.length - overlapTokens));
        tokens = buf.length;
      }
    }
    buf.push(line);
    tokens += addTokens;
  }
  flush();
  return chunks;
}

async function indexDir(dir = ROOT) {
  const files = fs.readdirSync(dir).filter(f => /\.(md|txt)$/i.test(f));
  let totalFiles = 0, totalChunks = 0;

  for (const file of files) {
    const p = path.join(dir, file);
    const raw = fs.readFileSync(p, 'utf8');
    const chunks = splitMarkdownSmart(raw);
    totalFiles++;
    totalChunks += chunks.length;
    console.log(`[${file}] ${chunks.length} chunk(s)`);

    // 🟢 envoi en batch
    for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
      const slice = chunks.slice(i, i + BATCH_SIZE)
        .map(text => ({ text, source: file }));
      await addDocuments(slice);
      if (chunks.length > BATCH_SIZE) {
        console.log(`  -> batch ${i + 1}-${Math.min(i + BATCH_SIZE, chunks.length)} / ${chunks.length}`);
      }
    }
  }

  console.log(`✅ Indexation terminée : ${totalFiles} fichier(s), ${totalChunks} chunk(s).`);
}

if (require.main === module) {
  indexDir().catch(e => {
    console.error('Indexation échouée:', e);
    process.exit(1);
  });
}

module.exports = { indexDir, splitMarkdownSmart };
