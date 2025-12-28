# -------------------------------
# Stage 1 : Build
# -------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Dépendances système nécessaires pour ChromaDB / build
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libstdc++6 \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copier requirements et installer
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# -------------------------------
# Stage 2 : Runtime léger
# -------------------------------
FROM python:3.11-slim

WORKDIR /app

# Copier uniquement le site-packages depuis builder
COPY --from=builder /install /usr/local

# Copier le code source
COPY . .

# Exposer port Streamlit
EXPOSE 8501

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Commande de démarrage
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
