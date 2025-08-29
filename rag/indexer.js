// rag/indexer.js
const fs = require('fs');
const path = require('path');
const { addDocuments } = require('./retriever');

const ROOT = path.resolve(__dirname, '..', 'knowledges');
const BATCH_SIZE = parseInt(process.env.INDEX_BATCH || '32', 10); // petits batchs = stabilité
const TARGET_MAX = parseInt(process.env.CHUNK_MAX_TOKENS || '420', 10);
const OVERLAP = parseInt(process.env.CHUNK_OVERLAP || '50', 10);
const SENTENCE_EXPLODE = process.env.SENTENCE_EXPLODE === '1'; // indexer aussi les phrases

function tokenizeCount(s) {
  return Math.max(1, Math.ceil(s.split(/\s+/).length * 0.9));
}

function splitIntoSentences(text) {
  // Segmentation simple mais robuste pour FR
  const parts = text
    .replace(/\n+/g, ' ')
    .split(/(?<=[\.\?\!…])\s+(?=[A-ZÀ-ÖØ-Ý0-9])/)
    .map(s => s.trim())
    .filter(Boolean);
  return parts;
}

function extractYear(text) {
  const m = text.match(/\b(19|20)\d{2}\b/);
  return m ? m[0] : null;
}

// Découpe Markdown en conservant le "chemin" de titres (H1 > H2 > H3)
function splitMarkdownSmart(text, {
  targetTokensMax = TARGET_MAX,
  overlapTokens = OVERLAP
} = {}) {
  const lines = text.split('\n');
  const chunks = [];
  let buf = [];
  let tokens = 0;
  let inFence = false;
  const hStack = []; // chemin de titres

  const flush = (reason = '') => {
    if (!buf.length) return;
    const body = buf.join('\n').trim();
    if (!body) { buf = []; tokens = 0; return; }
    const hpath = hStack.join(' > ');
    const year = extractYear(body);
    chunks.push({ text: body, hpath, year });
    buf = []; tokens = 0;
  };

  for (const raw of lines) {
    const line = raw;
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h && !inFence) {
      // nouveau titre -> flush précédent
      flush('heading');
      const level = h[1].length;
      hStack.splice(level - 1);
      hStack[level - 1] = h[2].trim();
    }
    if (line.trim().startsWith('```')) inFence = !inFence;

    const add = line + '\n';
    const addTokens = tokenizeCount(add);
    if (!inFence && tokens + addTokens > targetTokensMax) {
      flush('max');
      if (overlapTokens > 0) {
        const words = line.split(/\s+/);
        buf = words.slice(Math.max(0, words.length - overlapTokens));
        tokens = buf.length;
      }
    }
    buf.push(line);
    tokens += addTokens;
  }
  flush('end');
  return chunks;
}

async function indexDir(dir = ROOT) {
  const files = fs.readdirSync(dir).filter(f => /\.(md|txt)$/i.test(f));
  let totalFiles = 0, totalChunks = 0, totalDocs = 0;

  for (const file of files) {
    const p = path.join(dir, file);
    const raw = fs.readFileSync(p, 'utf8');
    const chunks = splitMarkdownSmart(raw);
    totalFiles++;
    totalChunks += chunks.length;
    console.log(`[${file}] ${chunks.length} chunk(s)`);

    for (const [idx, { text, hpath, year }] of chunks.entries()) {
      const docs = [];

      // 1) chunk principal avec hpath/date en tête
      const decorated = `${hpath ? `[${hpath}] ` : ''}${year ? `[${year}] ` : ''}${text}`;
      docs.push({ text: decorated, source: `${file}#${idx}` });

      // 2) optionnel : phrases (rappel ↑)
      if (SENTENCE_EXPLODE) {
        const sents = splitIntoSentences(text);
        sents.forEach((s, j) => {
          const st = `${hpath ? `[${hpath}] ` : ''}${year ? `[${year}] ` : ''}${s}`;
          docs.push({ text: st, source: `${file}#${idx}:s${j}` });
        });
      }

      // push par petits lots
      for (let i = 0; i < docs.length; i += BATCH_SIZE) {
        const slice = docs.slice(i, i + BATCH_SIZE);
        await addDocuments(slice);
        totalDocs += slice.length;
      }
    }
  }
  console.log(`Indexation terminée : ${totalFiles} fichier(s), ${totalChunks} chunk(s), ${totalDocs} doc(s).`);
}

if (require.main === module) {
  indexDir().catch(e => {
    console.error('Indexation échouée:', e);
    process.exit(1);
  });
}

module.exports = { indexDir, splitMarkdownSmart, splitIntoSentences };
