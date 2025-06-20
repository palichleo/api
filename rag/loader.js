// rag/loader.js
const fs = require('fs');
const path = require('path');

function loadMarkdown(filePath) {
  return fs.readFileSync(path.resolve(__dirname, '..', filePath), 'utf8');
}

module.exports = { loadMarkdown };