# Dockerfile

#image Python du départ
FROM python:3.11-slim

#dossier du travail dans le conteneur
WORKDIR /app

#Copie des dépendances et leur installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copie du code du projet dans le conteneur
COPY . .

#Commande pour lancer l'API FastAPI automatiquement
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]