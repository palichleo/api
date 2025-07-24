if docker ps -a --format '{{.Names}}' | grep -q '^chroma$'; then
  echo "Supprimer ancien conteneur Chroma..."
  docker rm -f chroma
fi

echo "Lancement de Chroma..."
docker run -d \
  --name chroma \
  --restart unless-stopped \
  -p 8000:8000 \
  -v chroma_data:/chroma/.chroma \
  ghcr.io/chroma-core/chroma:latest

echo "Chroma est lancé sur http://localhost:8000"