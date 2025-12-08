# Guide de Déploiement - Plateforme de Mobilité Intelligente d'Abidjan

Ce document fournit les instructions complètes pour déployer la plateforme.

## 📋 Prérequis

### Matériel Minimum
- **CPU**: 4 cores
- **RAM**: 16 GB (32 GB recommandé)
- **Stockage**: 100 GB SSD
- **Réseau**: 1 Gbps

### Logiciels
- Docker 24.0+
- Docker Compose 2.20+
- Python 3.10+
- Git

## 🚀 Déploiement

### 1. Cloner le Repository

```bash
git clone <repository-url>
cd abidjan_smart_city
```

### 2. Configuration

```bash
# Copier le fichier de configuration
cp config/.env.example config/.env

# Éditer les variables (modifier les mots de passe!)
nano config/.env
```

### 3. Démarrer l'Infrastructure

```bash
# Démarrer tous les services
docker-compose up -d

# Vérifier le statut
docker-compose ps

# Voir les logs
docker-compose logs -f
```

### 4. Initialiser les Bases de Données

```bash
# Attendre que PostgreSQL soit prêt
sleep 30

# Créer les schémas PostgreSQL
docker-compose exec postgres psql -U admin -d smart_city -f /docker-entrypoint-initdb.d/init.sql

# Ou exécuter le script Python
python -m src.db.init_postgres
python -m src.db.init_mongo
```

### 5. Créer les Topics Kafka

```bash
docker-compose exec kafka kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic traffic_data \
  --partitions 6 \
  --replication-factor 1

docker-compose exec kafka kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic weather_data \
  --partitions 3 \
  --replication-factor 1

docker-compose exec kafka kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic incident_alerts \
  --partitions 3 \
  --replication-factor 1
```

### 6. Démarrer l'API

```bash
# Installation des dépendances Python
pip install -r requirements.txt

# Démarrer l'API FastAPI
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 7. Démarrer les Producteurs de Données

```bash
# Dans des terminaux séparés:
python -m src.producers.traffic_producer
python -m src.producers.weather_producer
python -m src.producers.incident_producer
```

### 8. Configurer Grafana

1. Accéder à http://localhost:3000
2. Login: admin / admin
3. Ajouter la datasource PostgreSQL:
   - Host: postgres:5432
   - Database: smart_city
   - User: admin
4. Importer le dashboard: `dashboards/traffic_dashboard.json`

## 🔗 URLs des Services

| Service | URL | Credentials |
|---------|-----|-------------|
| API Docs | http://localhost:8000/docs | - |
| Grafana | http://localhost:3000 | admin/admin |
| Spark UI | http://localhost:8080 | - |

## 🧪 Tests

```bash
# Tests unitaires
pytest tests/ -v

# Tests de charge
locust -f tests/load_test.py --headless -u 100 -r 10 -t 60s
```

## 📊 Monitoring

```bash
# Vérifier les services Docker
docker-compose ps

# Logs en temps réel
docker-compose logs -f kafka
docker-compose logs -f spark-master
```

## 🛑 Arrêt

```bash
# Arrêter tous les services
docker-compose down

# Arrêter et supprimer les volumes
docker-compose down -v
```
