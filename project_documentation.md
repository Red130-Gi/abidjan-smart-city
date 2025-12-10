# Documentation Complète du Projet : Abidjan Smart City Traffic Platform

## 1. Vue d'Ensemble du Projet
**Objectif :** Développer une plateforme de gestion et de prédiction du trafic routier pour la ville d'Abidjan, capable d'optimiser les itinéraires et d'anticiper les congestions grâce à l'intelligence artificielle.

**Périmètre :**
*   Simulation de trafic réaliste.
*   Collecte et stockage de données en temps réel.
*   Analyse et prédiction via Machine Learning (XGBoost, LSTM, Ensemble).
*   Visualisation via Tableaux de bord interactifs (Grafana).
*   Optimisation d'itinéraires basée sur les prédictions.

---

## 2. Processus de Réalisation (Méthodologie)

Le projet a suivi une approche itérative et incrémentale :

### Phase 1 : Fondations & Simulation
*   Mise en place de l'environnement Docker.
*   Création du simulateur de trafic (`prediction_simulator`) générant des données réalistes (vitesse, densité) sur une topologie de graphe représentant Abidjan.
*   Configuration de la base de données PostgreSQL/TimescaleDB.

### Phase 2 : Analyse & Visualisation de Base
*   Développement des premiers tableaux de bord Grafana.
*   Visualisation des métriques en temps réel (Vitesse moyenne, Congestion).
*   Intégration d'une carte interactive Leaflet.

### Phase 3 : Intelligence Artificielle Avancée
*   Implémentation de modèles ML :
    *   **XGBoost :** Pour les prédictions à court terme et la performance rapide.
    *   **LSTM (Deep Learning) :** Pour capturer les séquences temporelles complexes à long terme.
    *   **Ensemble Model :** Fusion des deux modèles pour une précision optimale selon l'horizon de prédiction.
*   Développement d'un service de prédiction (`prediction_service`) et d'un pipeline d'entraînement continu.

### Phase 4 : Optimisation & Fonctionnalités Avancées
*   Création du `RouteOptimizer` utilisant l'algorithme A* pondéré par les prédictions de trafic.
*   Ajout de la détection d'anomalies.
*   Extension des dashboards avec des prédictions à 1h et des détails par segment.

---

## 3. Conception Technique (Architecture)

### 3.1 Stack Technologique
*   **Langage :** Python 3.9+
*   **Ingestion :** Apache Kafka
*   **Traitement :** Apache Spark (Streaming)
*   **Stockage Hybride :** 
    *   **Cassandra** (Données Brutes / Big Data)
    *   **PostgreSQL** (Données Agrégées / Serving)
*   **Visualisation :** Grafana
*   **Conteneurisation :** Docker & Docker Compose
*   **ML Libraries :** Scikit-learn, XGBoost, TensorFlow/Keras.

### 3.2 Architecture des Services (Big Data Hybride)
```mermaid
graph TD
    subgraph "Data Generation (Scale: 5000+ Vehicles)"
        TP[Traffic Producer] -->|JSON Stream| Kafka{Kafka Broker}
        WP[Weather Producer] -->|JSON Stream| Kafka
    end

    subgraph "Speed Layer (Real-Time)"
        Kafka -->|Subscribe| Spark[Spark Streaming]
        Spark -->|Raw Data (Write Heavy)| Cass[(Apache Cassandra)]
        Spark -->|Aggregated Stats| PG[(PostgreSQL)]
    end

    subgraph "Batch/ML Layer"
        Cass -->|Training Data| ML[ML Service (XGBoost/LSTM)]
        ML -->|Predictions| PG
    end

    subgraph "Serving Layer"
        PG -->|Query Stats/Preds| API[FastAPI]
        PG -->|Query Metrics| Grafana[Grafana Dashboard]
        API -->|Route Optimization| Client[Web/Mobile App]
    end
```

### 3.3 Modèle de Données
*   **`traffic_data` :** Données historiques brutes (segment_id, timestamp, speed, congestion_level).
*   **`traffic_predictions` :** Prédictions générées (model_type, horizon, predicted_speed, confidence).
*   **`predicted_anomalies` :** Événements anormaux détectés.

---

## 4. Réalisation & Fonctionnalités Clés

### 4.1 Dashboard de Supervision (Grafana)
*   **Vue Temps Réel :** Jauges de vitesse, compteurs de véhicules actifs.
*   **Carte Interactive :** Coloration des segments selon la congestion (Vert -> Rouge).
*   **Prédictions :**
    *   Graphiques temporels comparant Historique vs Prédiction.
    *   Focus sur l'horizon 1h avec le modèle d'Ensemble.
    *   Tableau détaillé des segments critiques.

### 4.2 Moteur de Prédiction Hybride
Le système utilise une stratégie de pondération dynamique :
*   **Horizon < 30 min :** Priorité au XGBoost (réactif).
*   **Horizon > 30 min :** Priorité au LSTM (tendanciel).
*   **Résultat :** Une robustesse accrue face aux variations soudaines et aux tendances journalières.

### 4.3 Optimiseur d'Itinéraire
L'algorithme de routing ne se base pas seulement sur la distance, mais sur le **temps de parcours prédit**.
*   *Exemple :* Si un embouteillage est prédit dans 20 min sur l'axe principal, le système déroutera l'usager par une voie secondaire fluide.

---

## 5. Déploiement & Opérations

Le projet est entièrement conteneurisé pour un déploiement "One-Click" :
```bash
docker-compose up --build
```
Les services redémarrent automatiquement (policy `always`) et les volumes Docker assurent la persistance des données (PostgreSQL, Grafana).
