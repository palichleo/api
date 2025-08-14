const { ChromaClient } = require('chromadb');

(async () => {
  const chroma = new ChromaClient({ host: 'localhost', port: 8000 });
  try {
    await chroma.deleteCollection({ name: 'leoknowledge' });
    console.log('Collection leoknowledge supprimée.');
  } catch (err) {
    console.error('Erreur suppression collection:', err.message);
  }
})();
