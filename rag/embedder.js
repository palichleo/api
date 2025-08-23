// rag/embedder.js - Version optimisée pour CPU
const { pipeline, env } = require('@xenova/transformers');

// Configuration pour CPU
env.allowRemoteModels = false;
env.localModelPath = process.env.MODEL_PATH || '/opt/models';
env.backends.onnx.wasm.numThreads = 4; // Ajuster selon votre CPU

let embedder;
let initPromise;

// Modèles légers pour CPU (ordre de préférence)
const MODELS = [
  'Xenova/all-MiniLM-L6-v2',      // 22M params, excellent rapport qualité/perf
  'Xenova/paraphrase-MiniLM-L3-v2', // Encore plus léger
  'Xenova/bge-small-en-v1.5',     // Alternative légère
  'Xenova/bge-m3'                 // Fallback (votre modèle actuel)
];

async function initEmbedder() {
  if (embedder) return embedder;
  if (initPromise) return initPromise;
  
  initPromise = (async () => {
    console.log('[EMBED] Initialisation du modèle d\'embedding...');
    
    for (const modelName of MODELS) {
      try {
        console.log(`[EMBED] Tentative avec ${modelName}...`);
        embedder = await pipeline('feature-extraction', modelName, {
          quantized: true,  // Utiliser la version quantifiée pour CPU
          progress_callback: (progress) => {
            if (progress.status === 'downloading') {
              console.log(`[EMBED] Téléchargement: ${Math.round(progress.progress)}%`);
            }
          }
        });
        console.log(`[EMBED] Modèle chargé: ${modelName}`);
        return embedder;
      } catch (e) {
        console.warn(`[EMBED] Échec ${modelName}:`, e.message);
      }
    }
    
    throw new Error('Aucun modèle d\'embedding disponible');
  })();
  
  return initPromise;
}

// Queue pour batching
const embeddingQueue = [];
let processingTimeout;
const BATCH_SIZE = 4;
const BATCH_DELAY = 50; // ms

async function processBatch() {
  if (embeddingQueue.length === 0) return;
  
  const batch = embeddingQueue.splice(0, BATCH_SIZE);
  const texts = batch.map(item => item.text);
  
  try {
    const model = await initEmbedder();
    console.log(`[EMBED] Traitement batch de ${texts.length} textes`);
    
    // Traiter le batch
    const outputs = await model(texts, { 
      pooling: 'mean', 
      normalize: true 
    });
    
    // Si sortie unique, la convertir en array
    const results = outputs.length === texts.length 
      ? outputs 
      : [outputs];
    
    // Résoudre les promesses
    batch.forEach((item, i) => {
      const embedding = Array.from(results[i]?.data || results.data);
      item.resolve(embedding);
    });
    
  } catch (error) {
    console.error('[EMBED] Erreur batch:', error);
    batch.forEach(item => item.reject(error));
  }
  
  // Traiter le prochain batch s'il y en a
  if (embeddingQueue.length > 0) {
    processingTimeout = setTimeout(processBatch, 10);
  }
}

// Fonction principale d'embedding avec batching
async function embed(text) {
  // Tronquer les textes trop longs pour CPU
  const maxLength = 256; // Réduire pour performance CPU
  const truncated = text.length > maxLength 
    ? text.substring(0, maxLength) + '...' 
    : text;
  
  return new Promise((resolve, reject) => {
    embeddingQueue.push({ 
      text: truncated, 
      resolve, 
      reject 
    });
    
    // Déclencher le traitement
    if (!processingTimeout) {
      processingTimeout = setTimeout(processBatch, BATCH_DELAY);
    }
    
    // Forcer le traitement si le batch est plein
    if (embeddingQueue.length >= BATCH_SIZE) {
      clearTimeout(processingTimeout);
      processingTimeout = setTimeout(processBatch, 0);
    }
  });
}

// Fonction pour embeddings multiples (plus efficace)
async function embedBatch(texts) {
  const model = await initEmbedder();
  
  // Tronquer tous les textes
  const maxLength = 256;
  const truncated = texts.map(t => 
    t.length > maxLength ? t.substring(0, maxLength) + '...' : t
  );
  
  console.log(`[EMBED] Batch de ${truncated.length} textes`);
  
  const output = await model(truncated, { 
    pooling: 'mean', 
    normalize: true 
  });
  
  // Gérer différents formats de sortie
  if (output.length === texts.length) {
    return output.map(o => Array.from(o.data));
  } else {
    // Sortie unique, dupliquer pour chaque texte
    const embedding = Array.from(output.data);
    return texts.map(() => embedding);
  }
}

// Préchauffage du modèle
async function warmup() {
  try {
    console.log('[EMBED] Préchauffage...');
    await initEmbedder();
    await embed('test warmup');
    console.log('[EMBED] Modèle prêt!');
  } catch (e) {
    console.error('[EMBED] Échec préchauffage:', e);
  }
}

// Lancer le warmup après 500ms
setTimeout(warmup, 500);

module.exports = { embed, embedBatch };