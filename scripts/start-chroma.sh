#!/bin/bash

if docker ps -a --format '{{.Names}}' | grep -q '^chroma$'; then
  echo "Suppression de l'ancien conteneur Chroma..."
  docker rm -f chroma
fi

echo "Lancement de Chroma dans Docker..."
docker run -d \
  --name chroma \
  --restart unless-stopped \
  -p 8000:8000 \
  -v chroma_data:/chroma/.chroma \
  ghcr.io/chroma-core/chroma:latest