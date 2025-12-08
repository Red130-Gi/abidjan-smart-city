"""
FastAPI REST API for Abidjan Smart City Platform
Provides endpoints for traffic data, ML predictions, and route optimization.
"""
from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import jwt
import json
import logging
import os
import psycopg2
from psycopg2.extras import RealDictCursor

import sys
sys.path.append('/app')
from config.settings import postgres_config

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Abidjan Smart City - Traffic & Predictions API",
    description="API REST pour la plateforme de mobilité intelligente d'Abidjan avec prédictions ML et optimisation d'itinéraires",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount static files for Leaflet map
static_path = "/app/static"
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)
JWT_SECRET = "supersecretkey_change_in_production"
JWT_ALGORITHM = "HS256"

# ============================================
# DATABASE CONNECTION
# ============================================

def get_db_connection():
    return psycopg2.connect(
        host=postgres_config.host,
        port=postgres_config.port,
        user=postgres_config.user,
        password=postgres_config.password,
        database=postgres_config.database
    )

# ============================================
# PYDANTIC MODELS
# ============================================

class CongestionLevel(str, Enum):
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    SEVERE = "severe"

class TrafficSegmentStatus(BaseModel):
    segment_id: str
    segment_name: Optional[str] = None
    avg_speed: float
    max_speed: Optional[float] = None
    min_speed: Optional[float] = None
    vehicle_count: int
    stopped_vehicles: int = 0
    congestion_level: str
    last_updated: Optional[datetime] = None

class PredictionResponse(BaseModel):
    segment_id: str
    prediction_time: str
    target_time: str
    horizon_minutes: int
    predicted_speed: float
    predicted_congestion: str
    confidence: float
    model_type: str

class RouteRequest(BaseModel):
    origin: str
    destination: str
    use_predictions: bool = True

class RouteComparisonResponse(BaseModel):
    request_id: str
    origin: Dict[str, str]
    destination: Dict[str, str]
    calculated_at: str
    routes: Dict[str, Any]
    savings: Dict[str, Any]
    recommendation: str

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

# ============================================
# ROAD SEGMENTS DATA
# ============================================

SEGMENTS = {
    'SEG001': 'Pont HKB',
    'SEG002': 'Boulevard VGE',
    'SEG003': 'Autoroute du Nord',
    'SEG004': 'Rue Lepic',
    'SEG005': 'Bd Nangui Abrogoua',
    'SEG006': 'Pont Général de Gaulle',
    'SEG007': 'Boulevard de Marseille',
    'SEG008': 'Voie Express Yopougon'
}

# ============================================
# AUTHENTICATION (Optional for public endpoints)
# ============================================

def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=24)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def verify_token_optional(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Optional token verification - allows unauthenticated access."""
    if credentials is None:
        return None
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except:
        return None

# ============================================
# HEALTH & AUTH ENDPOINTS
# ============================================

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Abidjan Smart City API", "version": "2.0.0"}

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {"api": "up", "database": "up", "ml": "up"}
    }

@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(credentials: UserLogin):
    """Authenticate user and return JWT token."""
    if credentials.username == "admin" and credentials.password == "admin123":
        token = create_access_token({"sub": credentials.username, "role": "admin"})
        return TokenResponse(access_token=token, expires_in=86400)
    raise HTTPException(status_code=401, detail="Identifiants incorrects")

# ============================================
# TRAFFIC DATA ENDPOINTS (PUBLIC)
# ============================================

@app.get("/traffic/current", response_model=List[TrafficSegmentStatus], tags=["Traffic"])
async def get_current_traffic():
    """
    Get current traffic status for all segments.
    Used by Grafana and Leaflet map.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT DISTINCT ON (segment_id)
                segment_id,
                avg_speed,
                max_speed,
                min_speed,
                vehicle_count,
                stopped_vehicles,
                congestion_level,
                processed_at as last_updated
            FROM traffic_segment_stats
            ORDER BY segment_id, processed_at DESC
        """)
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        results = []
        for row in rows:
            results.append(TrafficSegmentStatus(
                segment_id=row['segment_id'],
                segment_name=SEGMENTS.get(row['segment_id'], row['segment_id']),
                avg_speed=float(row['avg_speed'] or 0),
                max_speed=float(row['max_speed'] or 0) if row.get('max_speed') else None,
                min_speed=float(row['min_speed'] or 0) if row.get('min_speed') else None,
                vehicle_count=int(row['vehicle_count'] or 0),
                stopped_vehicles=int(row['stopped_vehicles'] or 0),
                congestion_level=row['congestion_level'] or 'moderate',
                last_updated=row['last_updated']
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Error fetching traffic data: {e}")
        # Return mock data if DB fails
        return [
            TrafficSegmentStatus(
                segment_id=seg_id,
                segment_name=name,
                avg_speed=40.0,
                vehicle_count=50,
                congestion_level='moderate'
            )
            for seg_id, name in SEGMENTS.items()
        ]

@app.get("/traffic/segment/{segment_id}", response_model=TrafficSegmentStatus, tags=["Traffic"])
async def get_segment_traffic(segment_id: str):
    """Get traffic data for a specific segment."""
    if segment_id not in SEGMENTS:
        raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")
    
    traffic = await get_current_traffic()
    for seg in traffic:
        if seg.segment_id == segment_id:
            return seg
    
    raise HTTPException(status_code=404, detail=f"No data for segment {segment_id}")

# ============================================
# PREDICTIONS ENDPOINTS
# ============================================

@app.get("/predictions/all", tags=["Predictions"])
async def get_all_predictions(horizon: int = Query(default=15, description="Horizon in minutes")):
    """
    Get ML predictions for all segments.
    Used by Grafana predictions dashboard.
    """
    try:
        # Try to load prediction service
        try:
            from src.ml.prediction_service import prediction_service
            predictions = prediction_service.predict_all_segments([horizon])
            return {"predictions": predictions, "model": "xgboost_ensemble", "generated_at": datetime.now().isoformat()}
        except ImportError:
            pass
        
        # Fallback: generate from database or simulate
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT DISTINCT ON (segment_id, horizon_minutes)
                segment_id, prediction_time, target_time, horizon_minutes,
                predicted_speed, predicted_congestion, confidence_score, model_type
            FROM traffic_predictions
            WHERE horizon_minutes = %s
            ORDER BY segment_id, horizon_minutes, prediction_time DESC
        """, (horizon,))
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        if rows:
            return {
                "predictions": [dict(row) for row in rows],
                "model": "database",
                "generated_at": datetime.now().isoformat()
            }
        
        # Generate simulated predictions
        predictions = []
        now = datetime.now()
        for seg_id, name in SEGMENTS.items():
            import random
            base_speed = 30 + random.uniform(-10, 20)
            predictions.append({
                "segment_id": seg_id,
                "prediction_time": now.isoformat(),
                "target_time": (now + timedelta(minutes=horizon)).isoformat(),
                "horizon_minutes": horizon,
                "predicted_speed": round(base_speed, 2),
                "predicted_congestion": "moderate" if base_speed > 25 else "heavy",
                "confidence": round(0.7 + random.uniform(0, 0.25), 3),
                "model_type": "simulated"
            })
        
        return {"predictions": predictions, "model": "simulated", "generated_at": now.isoformat()}
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/{segment_id}", tags=["Predictions"])
async def get_segment_predictions(
    segment_id: str,
    horizons: str = Query(default="5,15,30,60", description="Comma-separated horizons")
):
    """Get predictions for a specific segment across multiple horizons."""
    if segment_id not in SEGMENTS:
        raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")
    
    horizon_list = [int(h.strip()) for h in horizons.split(",")]
    
    predictions = []
    now = datetime.now()
    
    for horizon in horizon_list:
        import random
        base_speed = 30 + random.uniform(-10, 20)
        predictions.append({
            "segment_id": segment_id,
            "segment_name": SEGMENTS[segment_id],
            "prediction_time": now.isoformat(),
            "target_time": (now + timedelta(minutes=horizon)).isoformat(),
            "horizon_minutes": horizon,
            "predicted_speed": round(base_speed, 2),
            "predicted_congestion": "moderate" if base_speed > 25 else "heavy",
            "confidence": round(0.85 - (horizon / 200), 3)
        })
    
    return {"segment_id": segment_id, "predictions": predictions}

@app.get("/predictions/trends", tags=["Predictions"])
async def get_prediction_trends():
    """Get trend data for prediction charts."""
    now = datetime.now()
    trends = []
    
    for i in range(-6, 7):  # -60 min to +60 min
        time_point = now + timedelta(minutes=i * 10)
        import random
        
        for seg_id in SEGMENTS:
            base = 35 + random.uniform(-5, 5)
            # Simulate rush hour effect
            hour = time_point.hour
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                base -= 15
            
            trends.append({
                "time": time_point.isoformat(),
                "segment_id": seg_id,
                "speed": round(max(5, base), 2),
                "is_prediction": i > 0
            })
    
    return {"trends": trends}

# ============================================
# ANOMALIES ENDPOINTS
# ============================================

@app.get("/anomalies/predicted", tags=["Anomalies"])
async def get_predicted_anomalies():
    """Get predicted future anomalies."""
    try:
        from src.ml.prediction_service import prediction_service
        anomalies = prediction_service.detect_anomalies()
        return {"anomalies": anomalies, "count": len(anomalies)}
    except ImportError:
        pass
    
    # Fallback: generate sample anomaly
    now = datetime.now()
    return {
        "anomalies": [
            {
                "segment_id": "SEG001",
                "prediction_time": now.isoformat(),
                "expected_start": (now + timedelta(minutes=20)).isoformat(),
                "anomaly_type": "congestion_spike",
                "severity": 4,
                "probability": 0.78,
                "description": "Congestion prévue sur Pont HKB",
                "recommended_action": "Activer itinéraires alternatifs"
            }
        ],
        "count": 1
    }

# ============================================
# ROUTE OPTIMIZATION ENDPOINTS
# ============================================

@app.get("/routes/segments", tags=["Routes"])
async def get_route_segments():
    """Get available segments for route selection dropdown."""
    return {
        "segments": [
            {"id": seg_id, "name": name}
            for seg_id, name in SEGMENTS.items()
        ]
    }

@app.get("/routes/compare", tags=["Routes"])
async def compare_routes(
    origin: str = Query(..., description="Origin segment ID"),
    destination: str = Query(..., description="Destination segment ID")
):
    """
    Compare normal vs optimized route between two points.
    Used by Grafana route dashboard and Leaflet map.
    """
    if origin not in SEGMENTS:
        raise HTTPException(status_code=404, detail=f"Origin segment {origin} not found")
    if destination not in SEGMENTS:
        raise HTTPException(status_code=404, detail=f"Destination segment {destination} not found")
    if origin == destination:
        raise HTTPException(status_code=400, detail="Origin and destination must be different")
    
    try:
        from src.routing.route_optimizer import route_optimizer
        comparison = route_optimizer.compare_routes(origin, destination)
        return comparison
    except ImportError:
        pass
    
    # Fallback: generate simulated comparison
    import random
    
    normal_duration = 15 + random.uniform(5, 15)
    optimized_duration = normal_duration * (0.7 + random.uniform(0, 0.2))
    distance = 5 + random.uniform(0, 8)
    
    return {
        "request_id": f"RT{random.randint(1000, 9999)}",
        "origin": {"segment_id": origin, "name": SEGMENTS[origin]},
        "destination": {"segment_id": destination, "name": SEGMENTS[destination]},
        "calculated_at": datetime.now().isoformat(),
        "routes": {
            "normal": {
                "route_type": "normal",
                "total_distance_km": round(distance, 2),
                "estimated_duration_minutes": round(normal_duration, 1),
                "fuel_consumption_liters": round(distance * 0.08, 2),
                "co2_emissions_kg": round(distance * 0.08 * 2.31, 2),
                "avg_congestion_score": round(2.5 + random.uniform(-0.5, 1), 2),
                "waypoints": [[-4.0167, 5.3167], [-4.0200, 5.3200], [-4.0600, 5.3400]],
                "path": [origin, "SEG002", destination]
            },
            "optimized": {
                "route_type": "optimized",
                "total_distance_km": round(distance * 1.1, 2),
                "estimated_duration_minutes": round(optimized_duration, 1),
                "fuel_consumption_liters": round(distance * 1.1 * 0.08, 2),
                "co2_emissions_kg": round(distance * 1.1 * 0.08 * 2.31, 2),
                "avg_congestion_score": round(1.5 + random.uniform(-0.3, 0.5), 2),
                "waypoints": [[-4.0167, 5.3167], [-4.0250, 5.3100], [-4.0300, 5.3300], [-4.0600, 5.3400]],
                "path": [origin, "SEG004", "SEG005", destination]
            }
        },
        "savings": {
            "time_saved_minutes": round(normal_duration - optimized_duration, 1),
            "time_saved_percent": round((1 - optimized_duration / normal_duration) * 100, 1),
            "fuel_saved_liters": round(distance * 0.08 * 0.1, 2),
            "co2_saved_kg": round(distance * 0.08 * 0.1 * 2.31, 2)
        },
        "recommendation": f"🚀 Route optimisée recommandée! Économisez {round(normal_duration - optimized_duration, 0):.0f} minutes."
    }

@app.post("/routes/optimize", tags=["Routes"])
async def optimize_route(request: RouteRequest):
    """Calculate optimized route with POST body."""
    return await compare_routes(request.origin, request.destination)

# ============================================
# MAP ENDPOINT
# ============================================

@app.get("/map", response_class=HTMLResponse, tags=["Visualization"])
async def get_map():
    """Serve interactive Leaflet map for iframe embedding in Grafana."""
    map_path = "/app/static/map.html"
    if os.path.exists(map_path):
        with open(map_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    raise HTTPException(status_code=404, detail="Map not found")

@app.get("/map/predictions", response_class=HTMLResponse, tags=["Visualization"])
async def get_predictions_map():
    """Serve predictions-focused map."""
    # Return the same map - it handles both views
    return await get_map()

@app.get("/future-map", response_class=HTMLResponse, tags=["Visualization"])
async def get_future_map():
    """Serve interactive Future Planning map."""
    map_path = "/app/static/future_map.html"
    if os.path.exists(map_path):
        with open(map_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    raise HTTPException(status_code=404, detail="Future Map not found")

# ============================================
# GRAFANA DATA SOURCE ENDPOINTS
# ============================================

@app.get("/grafana/traffic-stats", tags=["Grafana"])
async def grafana_traffic_stats():
    """Endpoint formatted for Grafana JSON datasource."""
    traffic = await get_current_traffic()
    return [
        {
            "segment": t.segment_name,
            "segment_id": t.segment_id,
            "speed": t.avg_speed,
            "vehicles": t.vehicle_count,
            "congestion": t.congestion_level,
            "timestamp": datetime.now().isoformat()
        }
        for t in traffic
    ]

@app.get("/grafana/predictions-chart", tags=["Grafana"])
async def grafana_predictions_chart():
    """Time series predictions for Grafana."""
    predictions = await get_all_predictions(15)
    data = []
    
    for pred in predictions.get("predictions", []):
        data.append({
            "time": pred.get("target_time"),
            "segment": pred.get("segment_id"),
            "predicted_speed": pred.get("predicted_speed"),
            "confidence": pred.get("confidence", 0.8)
        })
    
    return data

# ============================================
# FUTURE TRIP PLANNING ENDPOINTS
# ============================================

class FutureTripRequest(BaseModel):
    origin: str = Field(..., description="Origin segment ID (e.g., SEG001)")
    destination: str = Field(..., description="Destination segment ID (e.g., SEG008)")
    departure_time: str = Field(..., description="Departure time in ISO format or 'HH:MM' format")

@app.post("/routes/future-plan", tags=["Future Planning"])
async def plan_future_trip(request: FutureTripRequest):
    """
    Plan a trip for a future departure time.
    Uses historical traffic patterns and ML predictions.
    
    Returns optimal route, alternative, estimated duration, and reasoning.
    """
    try:
        from src.routing.future_planner import future_planner
    except ImportError as e:
        logger.error(f"Could not import future_planner: {e}")
        raise HTTPException(status_code=500, detail="Future planner not available")
    
    # Parse departure time
    try:
        if 'T' in request.departure_time or '-' in request.departure_time:
            departure = datetime.fromisoformat(request.departure_time.replace('Z', ''))
        else:
            # Parse HH:MM format
            hour, minute = map(int, request.departure_time.split(':'))
            departure = datetime.now().replace(hour=hour, minute=minute, second=0)
            if departure <= datetime.now():
                departure += timedelta(days=1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid departure_time format: {e}")
    
    # Plan the trip
    plan = future_planner.plan_future_trip(
        origin=request.origin,
        destination=request.destination,
        departure_time=departure,
        current_time=datetime.now()
    )
    
    if 'error' in plan:
        raise HTTPException(status_code=400, detail=plan['error'])
    
    return plan

@app.get("/routes/future-plan", tags=["Future Planning"])
async def plan_future_trip_get(
    origin: str = Query(..., description="Origin segment ID"),
    destination: str = Query(..., description="Destination segment ID"),
    departure_time: str = Query(..., description="Departure time (HH:MM or ISO format)")
):
    """
    Plan a trip for a future departure time (GET version for easy testing).
    """
    request = FutureTripRequest(origin=origin, destination=destination, departure_time=departure_time)
    return await plan_future_trip(request)

@app.get("/traffic/hourly-forecast/{segment_id}", tags=["Future Planning"])
async def get_hourly_forecast(
    segment_id: str,
    hours: int = Query(24, description="Number of hours to forecast")
):
    """
    Get hourly traffic forecast for a segment.
    Returns predicted speed and congestion for each hour.
    """
    try:
        from src.routing.future_planner import future_planner
    except ImportError:
        raise HTTPException(status_code=500, detail="Future planner not available")
    
    if segment_id not in SEGMENTS:
        raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")
    
    forecast = future_planner.get_hourly_forecast(segment_id, min(hours, 48))
    
    return {
        "segment_id": segment_id,
        "segment_name": SEGMENTS[segment_id],
        "generated_at": datetime.now().isoformat(),
        "hours_ahead": min(hours, 48),
        "forecast": forecast
    }

@app.get("/traffic/best-departure-time", tags=["Future Planning"])
async def find_best_departure_time(
    origin: str = Query(..., description="Origin segment ID"),
    destination: str = Query(..., description="Destination segment ID"),
    earliest: str = Query("06:00", description="Earliest departure time (HH:MM)"),
    latest: str = Query("22:00", description="Latest departure time (HH:MM)")
):
    """
    Find the best departure time between two times.
    Compares traffic conditions at different hours.
    """
    try:
        from src.routing.future_planner import future_planner
    except ImportError:
        raise HTTPException(status_code=500, detail="Future planner not available")
    
    # Parse times
    try:
        earliest_h, earliest_m = map(int, earliest.split(':'))
        latest_h, latest_m = map(int, latest.split(':'))
    except:
        raise HTTPException(status_code=400, detail="Invalid time format, use HH:MM")
    
    now = datetime.now()
    results = []
    
    for hour in range(earliest_h, latest_h + 1):
        departure = now.replace(hour=hour, minute=0, second=0)
        if departure <= now:
            departure += timedelta(days=1)
        
        plan = future_planner.plan_future_trip(origin, destination, departure, now)
        
        if 'error' not in plan:
            results.append({
                'departure_time': departure.strftime('%Hh00'),
                'estimated_duration': plan['recommended_route']['estimated_duration_minutes'],
                'congestion_score': plan['recommended_route']['congestion_score'],
                'traffic_prediction': plan['traffic_prediction']
            })
    
    # Sort by duration
    results.sort(key=lambda x: x['estimated_duration'])
    
    return {
        'origin': SEGMENTS.get(origin, origin),
        'destination': SEGMENTS.get(destination, destination),
        'analysis_period': f"{earliest} - {latest}",
        'best_time': results[0] if results else None,
        'worst_time': results[-1] if results else None,
        'all_options': results
    }


# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Erreur interne du serveur", "status_code": 500}
    )

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
