# Plateforme de Mobilité Intelligente d'Abidjan (Smart City)

Ce projet est une implémentation complète d'une plateforme Big Data & IA pour la gestion de la mobilité urbaine à Abidjan.

## 📂 Structure du Projet

- **`ARCHITECTURE.md`** : Description détaillée de l'architecture technique (Lambda, Kafka, Spark, etc.).
- **`docker-compose.yml`** : Configuration de l'infrastructure conteneurisée.
- **`docs/`** : Documentation supplémentaire.
- **`src/`** : Code source (à venir).
    - **`producers/`** : Scripts Python pour la génération de données IoT.
    - **`spark/`** : Jobs Spark Streaming et Batch.
    - **`api/`** : API REST FastAPI.
    - **`ml/`** : Modèles de Machine Learning (XGBoost, LSTM).
- **`dashboards/`** : Configuration Grafana.

## 🚀 Démarrage Rapide

### Prérequis
- Docker & Docker Compose
- 8GB+ RAM recommandé

### Lancement de l'infrastructure
```bash
docker-compose up -d
```

Ceci démarrera :
- Kafka & Zookeeper
- Spark Master & Worker
- PostgreSQL (Port 5432)
- **Cassandra** (Port 9042 - Big Data)
- MongoDB (Port 27017)
- Redis (Port 6379)
- Grafana (Port 3000)

## 🛠 Stack Technique
- **Ingestion** : Apache Kafka
- **Traitement** : Apache Spark (PySpark)
- **Stockage** : 
  - **Cassandra** (Raw Data / Big Data - 5000+ véhicules)
  - **PostgreSQL** (PostGIS - Données Agrégées / Météo)
  - **MongoDB** (Logs)
  - **Redis** (Cache Temps Réel)
- **IA** : XGBoost, LSTM, TensorFlow/Keras + **Features Météo**
- **Backend** : FastAPI
- **Frontend/Viz** : Grafana
