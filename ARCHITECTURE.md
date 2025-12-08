# Architecture Globale - Plateforme de Mobilité Intelligente d'Abidjan

## 1. Vue d'ensemble
La plateforme repose sur une **Architecture Lambda**, conçue pour traiter des volumes massifs de données urbaines avec une latence minimale (< 1 seconde) tout en garantissant une précision historique. Elle combine une couche de vitesse (Speed Layer) pour le temps réel et une couche batch (Batch Layer) pour l'analyse approfondie et l'entraînement des modèles IA.

### Schéma de l'Architecture (Mermaid)

```mermaid
graph TD
    subgraph "Sources de Données (IoT)"
        IoT_GPS[Capteurs GPS / Gbakas]
        IoT_Traffic[Caméras Trafic]
        IoT_Weather[Données Météo]
    end

    subgraph "Ingestion (Kafka)"
        Kafka[Apache Kafka Cluster]
        Zookeeper[Zookeeper]
    end

    subgraph "Traitement (Spark)"
        SparkStream[Spark Streaming (Speed Layer)]
        SparkBatch[Spark Batch (Batch Layer)]
    end

    subgraph "Stockage (Hybrid)"
        Redis[(Redis - Cache Temps Réel)]
        Postgres[(PostgreSQL/PostGIS - Données Structurées)]
        Mongo[(MongoDB - Logs/Archives)]
    end

    subgraph "Intelligence Artificielle"
        Model_Traffic[Modèle Prédiction Trafic (XGBoost/LSTM)]
        Model_Anomaly[Détection Anomalies]
    end

    subgraph "Serving & API"
        FastAPI[API REST (FastAPI)]
        Auth[Auth Service (JWT)]
    end

    subgraph "Visualisation"
        Grafana[Grafana Dashboards]
    end

    IoT_GPS --> |JSON| Kafka
    IoT_Traffic --> |JSON| Kafka
    IoT_Weather --> |JSON| Kafka

    Kafka --> SparkStream
    Kafka --> SparkBatch

    SparkStream --> |Mise à jour état| Redis
    SparkStream --> |Alertes/Données| Postgres
    SparkBatch --> |Historique| Mongo
    SparkBatch --> |Entraînement| Model_Traffic

    Model_Traffic --> |Prédictions| SparkStream
    Model_Anomaly --> |Alertes| SparkStream

    FastAPI --> |Lecture| Redis
    FastAPI --> |Lecture| Postgres
    FastAPI --> |Lecture| Mongo

    Grafana --> |Visualisation| Postgres
    Grafana --> |Temps Réel| Redis
```

## 2. Composants Détaillés

### A. Ingestion (Kafka)
- **Rôle**: Tampon haute performance pour découpler les producteurs (capteurs) des consommateurs (Spark).
- **Configuration**: Cluster Kafka avec Zookeeper.
- **Topics**: `traffic_data`, `weather_data`, `incident_alerts`.

### B. Traitement (Spark Streaming & Batch)
- **Speed Layer (Spark Streaming)**: Traite les flux en temps réel. Calcule la vitesse moyenne, la densité du trafic, et détecte les anomalies instantanées. Latence cible < 1s.
- **Batch Layer (Spark Batch)**: Traite l'historique complet stocké dans MongoDB/HDFS (simulé ici par MongoDB) pour réentraîner les modèles et corriger les vues.

### C. Stockage Hybride
- **Redis**: Base de données In-Memory pour les données "chaudes" (état actuel du trafic, dernières alertes). Permet une lecture ultra-rapide pour l'API et les dashboards temps réel.
- **PostgreSQL + PostGIS**: Stockage relationnel et géospatial. Stocke la topologie de la ville (routes, ponts), les métadonnées des capteurs, et les agrégats structurés.
- **MongoDB**: Base NoSQL pour le stockage des données brutes (Data Lake) et des logs volumineux.

### D. Intelligence Artificielle
- **Prédiction de Trafic**: Ensemble Learning combinant XGBoost (pour les données tabulaires/contextuelles) et LSTM (pour les séquences temporelles).
- **Détection d'Anomalies**: Algorithmes statistiques (Z-score, IQR) et ML non supervisé (Isolation Forest) pour détecter les accidents ou congestions atypiques.

### E. API & Visualisation
- **FastAPI**: Expose les données via une API RESTful performante, asynchrone et documentée (Swagger). Sécurisée par JWT.
- **Grafana**: Tableaux de bord interactifs connectés à PostgreSQL et Redis pour visualiser le trafic, les alertes et les KPIs.

## 3. Flux de Données
1. **Collecte**: Les simulateurs IoT envoient des données de géolocalisation et de vitesse toutes les secondes vers Kafka.
2. **Ingestion**: Kafka reçoit les messages et les distribue dans les topics appropriés.
3. **Traitement Temps Réel**: Spark Streaming consomme les topics, enrichit les données (ajout météo, zone géographique), applique les modèles de prédiction, et met à jour Redis et PostgreSQL.
4. **Stockage & Batch**: Les données brutes sont sauvegardées dans MongoDB. Périodiquement, Spark Batch ré-entraîne les modèles.
5. **Consommation**: L'API FastAPI interroge Redis/Postgres pour servir les requêtes utilisateurs. Grafana affiche les métriques en temps réel.
