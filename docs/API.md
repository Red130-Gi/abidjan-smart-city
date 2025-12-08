# Documentation de l'API REST

## Base URL
```
http://localhost:8000/api/v1
```

## Authentication

L'API utilise JWT (JSON Web Tokens) pour l'authentification.

### Obtenir un Token

```http
POST /auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "admin123"
}
```

**Réponse:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### Utiliser le Token

```http
GET /api/v1/traffic/segments
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

---

## Endpoints

### Traffic

#### GET /traffic/segments
Récupère tous les segments routiers avec leur état actuel.

**Paramètres Query:**
- `congestion_level` (optionnel): Filtrer par niveau (light, moderate, heavy, severe)

**Réponse:**
```json
[
  {
    "segment_id": "SEG001",
    "segment_name": "Pont HKB",
    "avg_speed": 35.5,
    "max_speed": 48.2,
    "min_speed": 12.0,
    "vehicle_count": 120,
    "stopped_vehicles": 15,
    "congestion_level": "heavy",
    "last_updated": "2024-01-15T14:30:00Z"
  }
]
```

#### GET /traffic/segments/{segment_id}
Récupère l'état d'un segment spécifique.

#### GET /traffic/heatmap
Données pour la carte de chaleur du trafic.

---

### Predictions

#### GET /predictions/{segment_id}
Prédictions de trafic pour un segment.

**Paramètres Query:**
- `horizon_minutes` (défaut: 60, min: 5, max: 120)

**Réponse:**
```json
[
  {
    "segment_id": "SEG001",
    "segment_name": "Pont HKB",
    "target_time": "2024-01-15T15:00:00Z",
    "predicted_speed": 42.5,
    "predicted_congestion": "moderate",
    "confidence": 0.85,
    "model_version": "v2024.1"
  }
]
```

---

### Anomalies

#### GET /anomalies
Liste des anomalies détectées.

**Paramètres Query:**
- `severity_min` (défaut: 1, 1-5)
- `resolved` (optionnel): true/false
- `limit` (défaut: 20, max: 100)

**Réponse:**
```json
[
  {
    "anomaly_id": "ANO_20240115_001",
    "anomaly_type": "severe_congestion",
    "segment_id": "SEG001",
    "segment_name": "Pont HKB",
    "severity": 4,
    "confidence": 0.92,
    "detected_at": "2024-01-15T14:15:00Z",
    "details": {
      "avg_speed": 8.5,
      "vehicle_count": 150
    },
    "recommended_action": "Activer itinéraires alternatifs",
    "is_resolved": false
  }
]
```

---

### Incidents

#### GET /incidents
Liste des incidents en cours.

**Paramètres Query:**
- `active_only` (défaut: true)
- `incident_type` (optionnel): accident, breakdown, flooding, etc.

---

### Weather

#### GET /weather
Données météorologiques actuelles de toutes les stations.

---

### Statistics

#### GET /stats/summary
Résumé des statistiques globales.

**Réponse:**
```json
{
  "timestamp": "2024-01-15T14:30:00Z",
  "total_segments": 8,
  "average_city_speed": 42.5,
  "congestion_distribution": {
    "light": 2,
    "moderate": 4,
    "heavy": 1,
    "severe": 1
  },
  "active_incidents": 1,
  "active_anomalies": 3,
  "total_vehicles_tracked": 1250
}
```

---

## Codes d'Erreur

| Code | Description |
|------|-------------|
| 200 | Succès |
| 400 | Requête invalide |
| 401 | Non authentifié |
| 403 | Accès refusé |
| 404 | Ressource non trouvée |
| 429 | Rate limit dépassé |
| 500 | Erreur serveur |

## Rate Limiting

- 1000 requêtes par heure par clé API
- En-têtes de réponse:
  - `X-RateLimit-Limit`
  - `X-RateLimit-Remaining`
  - `X-RateLimit-Reset`
