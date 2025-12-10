"""
Future Route Planner - Predicts optimal routes for future departure times
Uses historical traffic patterns, temporal trends, and ML models to forecast traffic conditions.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import psycopg2
from dataclasses import dataclass
import json

import sys
sys.path.append('.')
sys.path.append('/app')
from config.settings import postgres_config

# Segments with coordinates
SEGMENTS = {
    'SEG001': {'name': 'Pont HKB', 'lat': 5.3167, 'lon': -4.0167},
    'SEG002': {'name': 'Boulevard VGE', 'lat': 5.3200, 'lon': -4.0200},
    'SEG003': {'name': 'Autoroute du Nord', 'lat': 5.3600, 'lon': -4.0100},
    'SEG004': {'name': 'Rue Lepic', 'lat': 5.3100, 'lon': -4.0250},
    'SEG005': {'name': 'Bd Nangui Abrogoua', 'lat': 5.3300, 'lon': -4.0300},
    'SEG006': {'name': 'Pont Général de Gaulle', 'lat': 5.3050, 'lon': -4.0150},
    'SEG007': {'name': 'Boulevard de Marseille', 'lat': 5.2950, 'lon': -4.0180},
    'SEG008': {'name': 'Voie Express Yopougon', 'lat': 5.3400, 'lon': -4.0600},
}

# Historical traffic patterns (average speeds by hour and day type)
HISTORICAL_PATTERNS = {
    # Hour: (weekday_speed, weekend_speed)
    0: (55, 60), 1: (58, 62), 2: (60, 63), 3: (60, 63), 4: (55, 60), 5: (45, 55),
    6: (35, 50), 7: (20, 45), 8: (15, 40), 9: (25, 42), 10: (35, 45), 11: (38, 48),
    12: (30, 45), 13: (32, 47), 14: (35, 48), 15: (32, 45), 16: (25, 42), 17: (15, 40),
    18: (18, 42), 19: (25, 45), 20: (35, 50), 21: (42, 52), 22: (48, 55), 23: (52, 58)
}

# Known recurring events that affect traffic
RECURRING_EVENTS = {
    'monday_morning': {'hours': [7, 8, 9], 'day': 0, 'impact': -15, 'reason': 'Début de semaine - forte affluence'},
    'friday_evening': {'hours': [16, 17, 18, 19], 'day': 4, 'impact': -20, 'reason': 'Départs week-end'},
    'market_day': {'hours': [8, 9, 10, 11], 'day': 2, 'impact': -10, 'reason': 'Jour de marché (Adjamé)'},
    'prayer_friday': {'hours': [12, 13, 14], 'day': 4, 'impact': -8, 'reason': 'Prière du vendredi'},
}

# Segment-specific patterns (some segments are more congested at certain times)
SEGMENT_PATTERNS = {
    'SEG001': {'peak_factor': 1.3, 'name': 'Pont HKB - Zone très congestionnée'},
    'SEG002': {'peak_factor': 1.2, 'name': 'Boulevard VGE - Zone commerciale'},
    'SEG003': {'peak_factor': 0.9, 'name': 'Autoroute du Nord - Voie rapide'},
    'SEG006': {'peak_factor': 1.4, 'name': 'Pont De Gaulle - Goulot d\'étranglement'},
    'SEG008': {'peak_factor': 0.85, 'name': 'Voie Express - Moins congestionnée'},
}

@dataclass
class FutureTrafficPrediction:
    """Prediction for a segment at a future time."""
    segment_id: str
    segment_name: str
    target_time: datetime
    predicted_speed: float
    congestion_level: str
    confidence: float
    reasons: List[str]

@dataclass
class FutureTripPlan:
    """Complete trip plan for a future departure."""
    origin: str
    destination: str
    departure_time: datetime
    current_time: datetime
    recommended_route: List[str]
    alternative_route: List[str]
    estimated_duration: float
    alternative_duration: float
    traffic_prediction: str
    congestion_score: float
    reasons: List[str]
    limitations: List[str]
    savings_vs_now: float


class FutureRoutePlanner:
    """
    Plans optimal routes for future departure times using historical patterns.
    """
    
    def __init__(self):
        self.conn = None
        self._load_historical_data()
    
    def _get_db_connection(self):
        """Get PostgreSQL connection."""
        return psycopg2.connect(
            host=postgres_config.host,
            port=postgres_config.port,
            user=postgres_config.user,
            password=postgres_config.password,
            database=postgres_config.database
        )
    
    def _load_historical_data(self):
        """Load historical traffic patterns from database."""
        try:
            conn = self._get_db_connection()
            
            # Get average speeds by hour and day of week
            query = """
                SELECT 
                    segment_id,
                    EXTRACT(HOUR FROM processed_at) as hour,
                    EXTRACT(DOW FROM processed_at) as day_of_week,
                    AVG(avg_speed) as avg_speed,
                    COUNT(*) as sample_count
                FROM traffic_segment_stats
                WHERE processed_at > NOW() - INTERVAL '7 days'
                GROUP BY segment_id, EXTRACT(HOUR FROM processed_at), EXTRACT(DOW FROM processed_at)
                ORDER BY segment_id, hour, day_of_week
            """
            
            self.historical_df = pd.read_sql(query, conn)
            conn.close()
            
            if len(self.historical_df) > 0:
                print(f"📊 Loaded {len(self.historical_df)} historical patterns")
            else:
                print("⚠️ No historical data, using default patterns")
                self.historical_df = None
                
        except Exception as e:
            print(f"Could not load historical data: {e}")
            self.historical_df = None
    
    def _get_historical_speed(self, segment_id: str, target_time: datetime) -> Tuple[float, List[str]]:
        """
        Get predicted speed for a segment at a specific future time.
        Uses historical data if available, otherwise uses default patterns.
        """
        hour = target_time.hour
        day_of_week = target_time.weekday()
        is_weekend = day_of_week >= 5
        reasons = []
        
        # Try to get from historical database
        if self.historical_df is not None and len(self.historical_df) > 0:
            mask = (
                (self.historical_df['segment_id'] == segment_id) & 
                (self.historical_df['hour'] == hour) & 
                (self.historical_df['day_of_week'] == day_of_week)
            )
            filtered = self.historical_df[mask]
            
            if len(filtered) > 0:
                speed = filtered['avg_speed'].values[0]
                reasons.append(f"Basé sur {int(filtered['sample_count'].values[0])} observations historiques")
                return float(speed), reasons
        
        # Use default patterns
        base_speed = HISTORICAL_PATTERNS[hour][1 if is_weekend else 0]
        
        # Apply segment-specific factors
        if segment_id in SEGMENT_PATTERNS:
            pattern = SEGMENT_PATTERNS[segment_id]
            # Peak hours have stronger effect
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                base_speed /= pattern['peak_factor']
                reasons.append(f"{pattern['name']}")
        
        # Check for recurring events
        for event_name, event in RECURRING_EVENTS.items():
            if hour in event['hours'] and day_of_week == event['day']:
                base_speed += event['impact']
                reasons.append(event['reason'])
        
        # Add time context
        if 7 <= hour <= 9:
            reasons.append("Heure de pointe du matin (7h-9h)")
        elif 17 <= hour <= 19:
            reasons.append("Heure de pointe du soir (17h-19h)")
        elif 22 <= hour or hour <= 5:
            reasons.append("Heures creuses (circulation fluide)")
        
        if is_weekend:
            reasons.append("Week-end (trafic réduit)")
        
        return max(10, min(70, base_speed)), reasons
    
    def _get_congestion_level(self, speed: float) -> str:
        """Convert speed to congestion level."""
        if speed < 15:
            return 'severe'
        elif speed < 25:
            return 'heavy'
        elif speed < 40:
            return 'moderate'
        else:
            return 'light'
    
    def predict_future_traffic(self, segment_id: str, target_time: datetime) -> FutureTrafficPrediction:
        """
        Predict traffic for a specific segment at a future time.
        """
        speed, reasons = self._get_historical_speed(segment_id, target_time)
        congestion = self._get_congestion_level(speed)
        
        # Confidence decreases with time horizon
        hours_ahead = (target_time - datetime.now()).total_seconds() / 3600
        confidence = max(0.5, 0.95 - (hours_ahead * 0.002))
        
        return FutureTrafficPrediction(
            segment_id=segment_id,
            segment_name=SEGMENTS.get(segment_id, {}).get('name', segment_id),
            target_time=target_time,
            predicted_speed=round(speed, 1),
            congestion_level=congestion,
            confidence=round(confidence, 2),
            reasons=reasons
        )
    
    def _calculate_route_duration(self, path: List[str], target_time: datetime) -> Tuple[float, float, List[str]]:
        """
        Calculate estimated duration for a route at a specific time.
        Returns: (duration_minutes, avg_congestion_score, reasons)
        """
        total_duration = 0
        total_congestion = 0
        all_reasons = []
        
        # Simplified distances between segments (km)
        segment_distances = {
            ('SEG001', 'SEG002'): 2.5, ('SEG002', 'SEG003'): 4.0,
            ('SEG002', 'SEG005'): 3.2, ('SEG001', 'SEG004'): 1.8,
            ('SEG004', 'SEG006'): 2.1, ('SEG006', 'SEG007'): 1.5,
            ('SEG005', 'SEG008'): 5.0, ('SEG003', 'SEG008'): 6.5,
            ('SEG001', 'SEG006'): 2.8, ('SEG004', 'SEG005'): 2.5,
        }
        
        congestion_scores = {'light': 1, 'moderate': 2, 'heavy': 3, 'severe': 4}
        
        for i in range(len(path) - 1):
            seg_from, seg_to = path[i], path[i + 1]
            
            # Get distance
            distance = segment_distances.get((seg_from, seg_to), 
                       segment_distances.get((seg_to, seg_from), 3.0))
            
            # Predict traffic for destination segment
            prediction = self.predict_future_traffic(seg_to, target_time)
            
            # Calculate time for this segment
            speed = max(10, prediction.predicted_speed)
            segment_time = (distance / speed) * 60  # minutes
            
            total_duration += segment_time
            total_congestion += congestion_scores.get(prediction.congestion_level, 2)
            
            # Collect unique reasons
            for reason in prediction.reasons:
                if reason not in all_reasons:
                    all_reasons.append(reason)
        
        avg_congestion = total_congestion / max(len(path) - 1, 1)
        
        return round(total_duration, 1), round(avg_congestion, 2), all_reasons
    
    def _find_routes(self, origin: str, destination: str) -> Tuple[List[str], List[str]]:
        """
        Find primary and alternative routes between two points.
        Returns: (primary_route, alternative_route)
        """
        # Define some predefined routes
        routes = {
            ('SEG001', 'SEG008'): [
                ['SEG001', 'SEG002', 'SEG005', 'SEG008'],  # Via Boulevard VGE
                ['SEG001', 'SEG002', 'SEG003', 'SEG008'],  # Via Autoroute du Nord
            ],
            ('SEG003', 'SEG007'): [
                ['SEG003', 'SEG002', 'SEG001', 'SEG006', 'SEG007'],
                ['SEG003', 'SEG008', 'SEG005', 'SEG004', 'SEG007'],
            ],
            ('SEG006', 'SEG003'): [
                ['SEG006', 'SEG001', 'SEG002', 'SEG003'],
                ['SEG006', 'SEG004', 'SEG005', 'SEG002', 'SEG003'],
            ],
        }
        
        # Try to find predefined routes
        key = (origin, destination)
        if key in routes:
            return routes[key][0], routes[key][1]
        
        # Reverse key
        key_rev = (destination, origin)
        if key_rev in routes:
            return list(reversed(routes[key_rev][0])), list(reversed(routes[key_rev][1]))
        
        # Generate simple routes
        primary = [origin, 'SEG002', destination]
        alternative = [origin, 'SEG005', destination]
        
        return primary, alternative
    
    def plan_future_trip(
        self, 
        origin: str, 
        destination: str, 
        departure_time: datetime,
        current_time: datetime = None
    ) -> Dict:
        """
        Plan an optimal route for a future departure time.
        
        Args:
            origin: Starting segment ID
            destination: Ending segment ID
            departure_time: When the user wants to leave
            current_time: Current time (for comparison)
            
        Returns:
            Complete trip plan with recommendations
        """
        if current_time is None:
            current_time = datetime.now()
        
        if origin not in SEGMENTS or destination not in SEGMENTS:
            return {'error': 'Invalid origin or destination segment'}
        
        if origin == destination:
            return {'error': 'Origin and destination must be different'}
        
        # Get routes
        primary_route, alt_route = self._find_routes(origin, destination)
        
        # Calculate durations at departure time
        primary_duration, primary_congestion, primary_reasons = self._calculate_route_duration(
            primary_route, departure_time
        )
        alt_duration, alt_congestion, alt_reasons = self._calculate_route_duration(
            alt_route, departure_time
        )
        
        # Calculate duration if leaving now (for comparison)
        now_duration, now_congestion, _ = self._calculate_route_duration(primary_route, current_time)
        
        # Determine recommended route
        if primary_duration <= alt_duration:
            recommended_route = primary_route
            recommended_duration = primary_duration
            alternative_route = alt_route
            alternative_duration = alt_duration
            reasons = primary_reasons
        else:
            recommended_route = alt_route
            recommended_duration = alt_duration
            alternative_route = primary_route
            alternative_duration = primary_duration
            reasons = alt_reasons
        
        # Overall traffic prediction
        if primary_congestion < 1.5:
            traffic_prediction = "🟢 Trafic fluide prévu"
        elif primary_congestion < 2.5:
            traffic_prediction = "🟡 Trafic modéré prévu"
        elif primary_congestion < 3.5:
            traffic_prediction = "🟠 Trafic dense prévu"
        else:
            traffic_prediction = "🔴 Trafic très dense prévu"
        
        # Limitations
        hours_ahead = (departure_time - current_time).total_seconds() / 3600
        limitations = [
            "Prédiction basée sur les tendances historiques",
            "Ne prend pas en compte les incidents imprévus (accidents, manifestations)",
            f"Horizon de {hours_ahead:.1f}h - précision réduite au-delà de 6h",
        ]
        
        if hours_ahead > 12:
            limitations.append("⚠️ Prédiction à long terme - faible fiabilité")
        
        # Build response
        result = {
            'request_id': f"FTP{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'origin': {
                'segment_id': origin,
                'name': SEGMENTS[origin]['name'],
                'coordinates': [SEGMENTS[origin]['lon'], SEGMENTS[origin]['lat']]
            },
            'destination': {
                'segment_id': destination,
                'name': SEGMENTS[destination]['name'],
                'coordinates': [SEGMENTS[destination]['lon'], SEGMENTS[destination]['lat']]
            },
            'timing': {
                'current_time': current_time.isoformat(),
                'departure_time': departure_time.isoformat(),
                'hours_until_departure': round(hours_ahead, 1),
                'departure_day': departure_time.strftime('%A'),
                'departure_hour': departure_time.strftime('%Hh%M')
            },
            'recommended_route': {
                'path': recommended_route,
                'path_names': [SEGMENTS[s]['name'] for s in recommended_route],
                'estimated_duration_minutes': recommended_duration,
                'congestion_score': primary_congestion,
                'waypoints': [[SEGMENTS[s]['lon'], SEGMENTS[s]['lat']] for s in recommended_route]
            },
            'alternative_route': {
                'path': alternative_route,
                'path_names': [SEGMENTS[s]['name'] for s in alternative_route],
                'estimated_duration_minutes': alternative_duration,
                'congestion_score': alt_congestion,
                'reason': "En cas d'imprévu sur la route principale",
                'waypoints': [[SEGMENTS[s]['lon'], SEGMENTS[s]['lat']] for s in alternative_route]
            },
            'traffic_prediction': traffic_prediction,
            'comparison': {
                'duration_if_leaving_now': now_duration,
                'duration_at_departure_time': recommended_duration,
                'difference_minutes': round(now_duration - recommended_duration, 1),
                'recommendation': self._get_timing_recommendation(now_duration, recommended_duration, hours_ahead)
            },
            'reasons': reasons,
            'limitations': limitations,
            'confidence': round(max(0.5, 0.95 - (hours_ahead * 0.02)), 2)
        }
        
        # Save to database
        self._save_trip_plan(result)
        
        return result
    
    def _get_timing_recommendation(self, now_duration: float, future_duration: float, hours_ahead: float) -> str:
        """Generate a recommendation about timing."""
        diff = now_duration - future_duration
        
        if diff > 10:
            return f"✅ Bon choix ! Partir dans {hours_ahead:.0f}h sera {diff:.0f} min plus rapide qu'actuellement"
        elif diff < -10:
            return f"⚠️ Attention : Partir maintenant serait {abs(diff):.0f} min plus rapide"
        else:
            return "ℹ️ Le temps de trajet sera similaire"
    
    def _save_trip_plan(self, plan: Dict):
        """Save trip plan to database for analytics."""
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO future_trip_plans 
                (request_id, origin_segment, destination_segment, departure_time,
                 estimated_duration, recommended_route, confidence, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT DO NOTHING
            """, (
                plan['request_id'],
                plan['origin']['segment_id'],
                plan['destination']['segment_id'],
                plan['timing']['departure_time'],
                plan['recommended_route']['estimated_duration_minutes'],
                json.dumps(plan['recommended_route']['path']),
                plan['confidence']
            ))
            
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Could not save trip plan: {e}")
    
    def get_hourly_forecast(self, segment_id: str, hours_ahead: int = 24) -> List[Dict]:
        """
        Get hourly traffic forecast for a segment.
        """
        forecasts = []
        now = datetime.now()
        
        for h in range(hours_ahead):
            target_time = now + timedelta(hours=h)
            prediction = self.predict_future_traffic(segment_id, target_time)
            
            forecasts.append({
                'hour': target_time.strftime('%Hh'),
                'datetime': target_time.isoformat(),
                'predicted_speed': prediction.predicted_speed,
                'congestion_level': prediction.congestion_level,
                'confidence': prediction.confidence
            })
        
        return forecasts


# Singleton instance
future_planner = FutureRoutePlanner()


if __name__ == "__main__":
    print("=" * 60)
    print("🔮 Future Route Planner Test")
    print("=" * 60)
    
    planner = FutureRoutePlanner()
    
    # Test: Plan a trip for 10h when it's currently earlier
    current = datetime.now()
    departure = current.replace(hour=10, minute=0, second=0)
    
    if departure <= current:
        departure += timedelta(days=1)
    
    print(f"\n📍 Current time: {current.strftime('%H:%M')}")
    print(f"🚀 Departure time: {departure.strftime('%H:%M %A')}")
    
    plan = planner.plan_future_trip(
        origin='SEG003',
        destination='SEG007',
        departure_time=departure,
        current_time=current
    )
    
    print(f"\n🗺️ From: {plan['origin']['name']}")
    print(f"🏁 To: {plan['destination']['name']}")
    print(f"\n{plan['traffic_prediction']}")
    print(f"\n⏱️ Estimated duration: {plan['recommended_route']['estimated_duration_minutes']} min")
    print(f"📍 Route: {' → '.join(plan['recommended_route']['path_names'])}")
    print(f"\n📋 Reasons:")
    for reason in plan['reasons']:
        print(f"   • {reason}")
    print(f"\n⚠️ Limitations:")
    for limit in plan['limitations']:
        print(f"   • {limit}")
