# Lancement du conteneur Docker
## A faire à chaque fois
Dans le répertoire racine :
docker-compose up -d

## A faire une seule fois
docker exec -it ollama-service ollama pull llama3.2:1b
docker exec -it ollama-service ollama pull nomic-embed-text
