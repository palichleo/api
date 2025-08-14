// rag/indexer.js

const fs = require('fs');
const path = require('path');
const { addDocument } = require('./retriever');

const KNOWLEDGE_DIR = path.join(__dirname, './knowledges'); // ⚠️ dossier "knowledges"

// --- utils comptage "tokens" approximé (mots) ---
function countTokens(text) {
  return text.split(/\s+/).filter(Boolean).length;
}

// Conserve les sections markdown par titres, sans casser les code-blocks.
function splitMarkdownSmart(text, {
  targetTokensMin = 600,
  targetTokensMax = 900,
  overlapTokens = 100
} = {}) {

  const lines = text.replace(/\r\n/g, '\n').split('\n');

  // 1) Marquage code fences pour éviter de splitter dedans
  const isFenceLine = (line) => /^```/.test(line.trim());
  let inCode = false;

  // 2) Repère les headers (1..6)
  const headerRe = /^(#{1,6})\s+(.*)$/;

  // 3) Construire des "sections" à partir des headers (en conservant le niveau)
  const sections = [];
  let current = [];
  let currentHeader = null;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (isFenceLine(line)) {
      inCode = !inCode;
      current.push(line);
      continue;
    }

    const m = !inCode && headerRe.exec(line);
    if (m) {
      // nouvelle section si on avait déjà du contenu
      if (current.length > 0) {
        sections.push({ header: currentHeader, content: current.join('\n') });
      }
      current = [line];
      currentHeader = { level: m[1].length, title: m[2].trim() };
    } else {
      current.push(line);
    }
  }
  if (current.length > 0) {
    sections.push({ header: currentHeader, content: current.join('\n') });
  }

  // 4) Rechunk des sections en paquets 700–900 tokens, overlap ~100
  const chunks = [];
  for (const sec of sections) {
    const raw = sec.content.trim();
    if (!raw) continue;

    const words = raw.split(/\s+/);
    let start = 0;
    const tokens = words.length;

    while (start < tokens) {
      const end = Math.min(start + targetTokensMax, tokens);

      // Ajuster pour ne pas couper au milieu d’un code-block
      // Heuristique simple: si le segment commence dans un code-block non clos, on étend jusqu’à la fin du fence suivant.
      let segment = words.slice(start, end).join(' ');
      // si on a un nombre impair de ``` on étend jusqu’à clôture dans la suite
      const fenceCount = (segment.match(/```/g) || []).length;
      if (fenceCount % 2 === 1 && end < tokens) {
        // cherche la clôture
        let j = end;
        while (j < tokens) {
          segment = words.slice(start, j).join(' ');
          const fc = (segment.match(/```/g) || []).length;
          if (fc % 2 === 0) {
            break;
          }
          j += 50; // étend par pas de 50 mots
        }
      }

      // Injecte un en‑tête "virtuel" pour conserver l’identité de la section
      const headerLine = sec.header
        ? `${'#'.repeat(sec.header.level)} ${sec.header.title}\n\n`
        : '';

      const finalText = headerLine + segment.trim();
      if (countTokens(finalText) >= Math.max(50, targetTokensMin) || chunks.length === 0) {
        chunks.push(finalText);
      }

      if (end >= tokens) break;
      // Overlap vers la fenêtre suivante
      start = Math.max(end - overlapTokens, start + 1);
    }
  }

  // Filtrer les micro‑chunks
  return chunks.filter(c => c.length > 30);
}

// --- main ---
async function indexKnowledge() {
  const files = fs.readdirSync(KNOWLEDGE_DIR).filter(f =>
    f.endsWith('.md') || f.endsWith('.txt')
  );

  let total = 0;
  for (const file of files) {
    const content = fs.readFileSync(path.join(KNOWLEDGE_DIR, file), 'utf8');
    const chunks = splitMarkdownSmart(content, {
      targetTokensMin: 600,
      targetTokensMax: 900,
      overlapTokens: 100
    });

    for (const chunk of chunks) {
      total++;
      console.log(`[${file}] →`, chunk.slice(0, 90).replace(/\n/g, ' ') + '...');
      await addDocument(chunk, file);
    }
  }

  console.log(`${files.length} fichier(s) traités, ${total} chunk(s) indexés depuis knowledges/`);
}

indexKnowledge().catch(err => {
  console.error('Erreur lors de l’indexation :', err);
});