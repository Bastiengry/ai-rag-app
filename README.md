# Créer l'image Docker

docker compose build --no-cache

# Lancer le conteneur Docker

## A faire à chaque fois

Dans le répertoire racine :
docker compose up -d

## A faire une seule fois

docker exec -it ollama-service ollama pull llama3.2:1b
**OU**
docker exec -it ollama-service ollama pull llama3.2:3b
**OU**
docker exec -it ollama-service ollama pull llama3.1:8b

docker exec -it ollama-service ollama pull nomic-embed-text

# Arrêter le conteneur Docker

## A faire à chaque fois

Dans le répertoire racine :
docker compose down

# Supprimer la BDD

- Supprimer le répertoire chroma_db
- Supprimer le répertoire parent_store
