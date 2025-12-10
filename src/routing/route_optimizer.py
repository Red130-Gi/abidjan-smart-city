"""
Route Optimization Engine using A* Algorithm
Calculates optimal routes based on real-time traffic and predictions.
Now with OSRM integration for real road geometry.
"""
import heapq
import json
import math
import uuid
import requests
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache

import psycopg2
from psycopg2.extras import execute_values
import sys
sys.path.append('.')
from config.settings import postgres_config


def decode_polyline(polyline_str: str, precision: int = 5) -> List[List[float]]:
    """
    Decode a Google/OSRM encoded polyline string into a list of coordinates.
    Returns: List of [longitude, latitude] pairs (GeoJSON order).
    """
    coordinates = []
    index = 0
    lat = 0
    lng = 0
    
    while index < len(polyline_str):
        # Decode latitude
        shift = 0
        result = 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if result & 1 else result >> 1
        lat += dlat
        
        # Decode longitude
        shift = 0
        result = 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if result & 1 else result >> 1
        lng += dlng
        
        # Convert to actual coordinates (GeoJSON format: [lng, lat])
        coordinates.append([lng / (10 ** precision), lat / (10 ** precision)])
    
    return coordinates

@dataclass
class Node:
    """Represents a road segment node in the graph."""
    segment_id: str
    name: str
    latitude: float
    longitude: float
    
@dataclass
class Edge:
    """Represents a connection between two segments."""
    from_node: str
    to_node: str
    distance_km: float
    base_duration_minutes: float
    current_speed: float = 40.0
    predicted_speed: float = 40.0
    congestion_level: str = 'light'

class RouteOptimizer:
    """
    A* based route optimizer with dynamic costs from traffic predictions.
    """
    
    # Abidjan road network (simplified graph)
    ROAD_NETWORK = {
        'nodes': {
            'SEG001': Node('SEG001', 'Pont HKB', 5.3167, -4.0167),
            'SEG002': Node('SEG002', 'Boulevard VGE', 5.3200, -4.0200),
            'SEG003': Node('SEG003', 'Autoroute du Nord', 5.3600, -4.0100),
            'SEG004': Node('SEG004', 'Rue Lepic', 5.3100, -4.0250),
            'SEG005': Node('SEG005', 'Bd Nangui Abrogoua', 5.3300, -4.0300),
            'SEG006': Node('SEG006', 'Pont Général de Gaulle', 5.3050, -4.0150),
            'SEG007': Node('SEG007', 'Boulevard de Marseille', 5.2950, -4.0180),
            'SEG008': Node('SEG008', 'Voie Express Yopougon', 5.3400, -4.0600),
        },
        'edges': [
            # Main connections (bidirectional)
            ('SEG001', 'SEG002', 2.5),
            ('SEG002', 'SEG003', 4.0),
            ('SEG002', 'SEG005', 3.2),
            ('SEG001', 'SEG004', 1.8),
            ('SEG004', 'SEG006', 2.1),
            ('SEG006', 'SEG007', 1.5),
            ('SEG005', 'SEG008', 5.0),
            ('SEG003', 'SEG008', 6.5),
            ('SEG001', 'SEG006', 2.8),
            ('SEG004', 'SEG005', 2.5),
            ('SEG007', 'SEG004', 1.9),
            ('SEG002', 'SEG004', 2.0),
        ]
    }
    
    CONGESTION_MULTIPLIERS = {
        'light': 1.0,
        'moderate': 1.3,
        'heavy': 1.8,
        'severe': 2.5
    }
    
    # OSRM API for real road geometry
    OSRM_URL = "http://router.project-osrm.org/route/v1/driving"
    
    def __init__(self):
        self.graph = self._build_graph()
        self._geometry_cache = {}  # Cache for OSRM geometries
        
    def _fetch_osrm_geometry(self, origin_coords: Tuple[float, float], 
                              dest_coords: Tuple[float, float]) -> Optional[List[List[float]]]:
        """
        Fetch real road geometry from OSRM API.
        
        Args:
            origin_coords: (longitude, latitude) of origin
            dest_coords: (longitude, latitude) of destination
            
        Returns:
            List of [lng, lat] coordinates forming the route, or None on failure.
        """
        cache_key = f"{origin_coords[0]:.4f},{origin_coords[1]:.4f}_{dest_coords[0]:.4f},{dest_coords[1]:.4f}"
        
        if cache_key in self._geometry_cache:
            return self._geometry_cache[cache_key]
        
        try:
            url = f"{self.OSRM_URL}/{origin_coords[0]},{origin_coords[1]};{dest_coords[0]},{dest_coords[1]}"
            params = {
                'overview': 'full',       # Get full geometry
                'geometries': 'polyline', # Encoded polyline format
                'steps': 'false'
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 'Ok' and data.get('routes'):
                    polyline = data['routes'][0]['geometry']
                    coordinates = decode_polyline(polyline)
                    self._geometry_cache[cache_key] = coordinates
                    return coordinates
        except Exception as e:
            print(f"⚠️ OSRM fetch failed: {e}")
        
        return None
        
    def _get_db_connection(self):
        return psycopg2.connect(
            host=postgres_config.host,
            port=postgres_config.port,
            user=postgres_config.user,
            password=postgres_config.password,
            database=postgres_config.database
        )
    
    def _build_graph(self) -> Dict[str, List[Edge]]:
        """Build adjacency list from road network definition."""
        graph = {node_id: [] for node_id in self.ROAD_NETWORK['nodes']}
        
        for edge_def in self.ROAD_NETWORK['edges']:
            from_node, to_node, distance = edge_def
            base_duration = distance / 40 * 60  # At 40 km/h base speed
            
            # Add both directions
            graph[from_node].append(Edge(
                from_node=from_node,
                to_node=to_node,
                distance_km=distance,
                base_duration_minutes=base_duration
            ))
            graph[to_node].append(Edge(
                from_node=to_node,
                to_node=from_node,
                distance_km=distance,
                base_duration_minutes=base_duration
            ))
        
        return graph
    
    def _update_edge_costs(self):
        """Update edge costs with current/predicted traffic data."""
        conn = self._get_db_connection()
        
        # Get current traffic conditions
        query = """
            SELECT DISTINCT ON (segment_id)
                segment_id, avg_speed, congestion_level
            FROM traffic_segment_stats
            ORDER BY segment_id, window_start DESC
        """
        
        try:
            cur = conn.cursor()
            cur.execute(query)
            current_data = {row[0]: {'speed': row[1], 'congestion': row[2]} 
                          for row in cur.fetchall()}
            cur.close()
            
            # Get predictions
            pred_query = """
                SELECT DISTINCT ON (segment_id)
                    segment_id, predicted_speed, predicted_congestion
                FROM traffic_predictions
                WHERE horizon_minutes = 15
                ORDER BY segment_id, 
                         CASE WHEN model_type = 'ensemble_lstm_xgb' THEN 1 ELSE 2 END,
                         prediction_time DESC
            """
            cur = conn.cursor()
            cur.execute(pred_query)
            predictions = {row[0]: {'speed': row[1], 'congestion': row[2]} 
                          for row in cur.fetchall()}
            cur.close()
            conn.close()
            
            # Update graph edges
            for node_id, edges in self.graph.items():
                for edge in edges:
                    seg_data = current_data.get(edge.to_node, {'speed': 40, 'congestion': 'light'})
                    pred_data = predictions.get(edge.to_node, seg_data)
                    
                    edge.current_speed = float(seg_data['speed'] or 40)
                    edge.predicted_speed = float(pred_data['speed'] or 40)
                    edge.congestion_level = pred_data['congestion'] or 'light'
        except Exception as e:
            print(f"Warning: Could not update edge costs: {e}")
            conn.close()
    
    def _heuristic(self, node_a: str, node_b: str) -> float:
        """Haversine distance heuristic for A*."""
        n1 = self.ROAD_NETWORK['nodes'][node_a]
        n2 = self.ROAD_NETWORK['nodes'][node_b]
        
        R = 6371  # Earth radius in km
        lat1, lon1 = math.radians(n1.latitude), math.radians(n1.longitude)
        lat2, lon2 = math.radians(n2.latitude), math.radians(n2.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c  # Distance in km
    
    def _get_edge_cost(self, edge: Edge, use_predictions: bool = False) -> float:
        """
        Calculate traversal cost for an edge (in minutes).
        
        For optimized routes (use_predictions=True):
          - Use the BETTER of current vs predicted speed (assume we take optimal timing)
          - Lower congestion multiplier (we avoid peak congestion)
        
        For normal routes (use_predictions=False):
          - Use current speed with full congestion impact
        """
        if use_predictions:
            # Optimized route: use best available speed and reduced congestion
            speed = max(edge.current_speed, edge.predicted_speed, 30)  # At least 30 km/h on optimized
            congestion_mult = 1.0  # No congestion penalty on optimized route
        else:
            # Normal route: use current speed with congestion
            speed = edge.current_speed
            speed = max(5, speed)  # Minimum 5 km/h
            congestion_mult = self.CONGESTION_MULTIPLIERS.get(edge.congestion_level, 1.0)
        
        base_time = edge.distance_km / speed * 60
        return base_time * congestion_mult
    
    def find_route(self, origin: str, destination: str, 
                   use_predictions: bool = False) -> Optional[Dict]:
        """
        Find optimal route using A* algorithm.
        
        Args:
            origin: Starting segment ID
            destination: Ending segment ID
            use_predictions: Use predicted speeds instead of current
            
        Returns:
            Route dictionary with path, distance, duration, etc.
        """
        if origin not in self.graph or destination not in self.graph:
            return None
        
        # Update edge costs with latest data
        self._update_edge_costs()
        
        # A* algorithm
        open_set = [(0, origin)]
        came_from = {}
        g_score = {origin: 0}
        f_score = {origin: self._heuristic(origin, destination)}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == destination:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                return self._build_route_result(path, use_predictions)
            
            for edge in self.graph[current]:
                neighbor = edge.to_node
                tentative_g = g_score[current] + self._get_edge_cost(edge, use_predictions)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, destination)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def _build_route_result(self, path: List[str], use_predictions: bool) -> Dict:
        """Build detailed route result from path, with OSRM real geometry."""
        segments = []
        total_distance = 0
        total_duration = 0
        total_duration_no_opt = 0
        total_congestion_score = 0
        
        waypoints = []
        
        for i, segment_id in enumerate(path):
            node = self.ROAD_NETWORK['nodes'][segment_id]
            waypoints.append([node.longitude, node.latitude])
            
            if i < len(path) - 1:
                next_seg = path[i + 1]
                edge = next((e for e in self.graph[segment_id] if e.to_node == next_seg), None)
                
                if edge:
                    cost = self._get_edge_cost(edge, use_predictions)
                    cost_normal = self._get_edge_cost(edge, False)
                    
                    total_distance += edge.distance_km
                    total_duration += cost
                    total_duration_no_opt += cost_normal
                    
                    cong_score = {'light': 1, 'moderate': 2, 'heavy': 3, 'severe': 4}
                    total_congestion_score += cong_score.get(edge.congestion_level, 2)
                    
                    segments.append({
                        'segment_order': i,
                        'segment_id': segment_id,
                        'segment_name': node.name,
                        'distance_km': round(edge.distance_km, 2),
                        'duration_minutes': round(cost, 1),
                        'current_speed': round(edge.current_speed, 1),
                        'predicted_speed': round(edge.predicted_speed, 1),
                        'congestion_level': edge.congestion_level,
                        'geometry': {
                            'type': 'Point',
                            'coordinates': [node.longitude, node.latitude]
                        }
                    })
        
        # Add final segment
        final_node = self.ROAD_NETWORK['nodes'][path[-1]]
        segments.append({
            'segment_order': len(path) - 1,
            'segment_id': path[-1],
            'segment_name': final_node.name,
            'distance_km': 0,
            'duration_minutes': 0,
            'congestion_level': 'destination'
        })
        
        avg_congestion = total_congestion_score / max(len(path) - 1, 1)
        fuel_consumption = total_distance * 0.08  # 8L/100km average
        co2_emissions = fuel_consumption * 2.31  # 2.31 kg CO2 per liter
        
        # Fetch real road geometry from OSRM (origin to destination)
        origin_node = self.ROAD_NETWORK['nodes'][path[0]]
        dest_node = self.ROAD_NETWORK['nodes'][path[-1]]
        osrm_geometry = self._fetch_osrm_geometry(
            (origin_node.longitude, origin_node.latitude),
            (dest_node.longitude, dest_node.latitude)
        )
        
        # Use OSRM geometry if available, otherwise fallback to waypoints
        route_coordinates = osrm_geometry if osrm_geometry else waypoints
        
        return {
            'path': path,
            'segments': segments,
            'total_distance_km': round(total_distance, 2),
            'estimated_duration_minutes': round(total_duration, 1),
            'duration_without_optimization': round(total_duration_no_opt, 1),
            'time_saved_minutes': round(total_duration_no_opt - total_duration, 1),
            'avg_congestion_score': round(avg_congestion, 2),
            'fuel_consumption_liters': round(fuel_consumption, 2),
            'co2_emissions_kg': round(co2_emissions, 2),
            'waypoints': waypoints,
            'route_geometry': {
                'type': 'LineString',
                'coordinates': route_coordinates
            },
            'geometry_source': 'osrm' if osrm_geometry else 'straight_line',
            'used_predictions': use_predictions
        }
    
    def find_alternative_routes(self, origin: str, destination: str, 
                                max_alternatives: int = 2) -> List[Dict]:
        """
        Find multiple alternative routes.
        Returns: List of routes sorted by duration
        """
        routes = []
        
        # Main optimal route with predictions
        optimal = self.find_route(origin, destination, use_predictions=True)
        if optimal:
            optimal['route_type'] = 'optimized'
            optimal['route_id'] = str(uuid.uuid4())[:8]
            routes.append(optimal)
        
        # Normal route without predictions
        normal = self.find_route(origin, destination, use_predictions=False)
        if normal:
            normal['route_type'] = 'normal'
            normal['route_id'] = str(uuid.uuid4())[:8]
            routes.append(normal)
        
        return sorted(routes, key=lambda r: r['estimated_duration_minutes'])
    
    def compare_routes(self, origin: str, destination: str) -> Dict:
        """
        Compare normal vs optimized route and provide analysis.
        """
        routes = self.find_alternative_routes(origin, destination)
        
        if len(routes) < 2:
            return {'error': 'Could not calculate routes'}
        
        optimized = next((r for r in routes if r['route_type'] == 'optimized'), routes[0])
        normal = next((r for r in routes if r['route_type'] == 'normal'), routes[1])
        
        origin_name = self.ROAD_NETWORK['nodes'][origin].name
        dest_name = self.ROAD_NETWORK['nodes'][destination].name
        
        request_id = str(uuid.uuid4())[:12]
        
        comparison = {
            'request_id': request_id,
            'origin': {'segment_id': origin, 'name': origin_name},
            'destination': {'segment_id': destination, 'name': dest_name},
            'calculated_at': datetime.now().isoformat(),
            'routes': {
                'normal': normal,
                'optimized': optimized
            },
            'savings': {
                'time_saved_minutes': round(
                    normal['estimated_duration_minutes'] - optimized['estimated_duration_minutes'], 1
                ),
                'time_saved_percent': round(
                    (1 - optimized['estimated_duration_minutes'] / normal['estimated_duration_minutes']) * 100, 1
                ) if normal['estimated_duration_minutes'] > 0 else 0,
                'fuel_saved_liters': round(
                    normal['fuel_consumption_liters'] - optimized['fuel_consumption_liters'], 2
                ),
                'co2_saved_kg': round(
                    normal['co2_emissions_kg'] - optimized['co2_emissions_kg'], 2
                )
            },
            'recommendation': self._generate_recommendation(normal, optimized)
        }
        
        return comparison
    
    def _generate_recommendation(self, normal: Dict, optimized: Dict) -> str:
        """Generate human-readable route recommendation."""
        time_diff = normal['estimated_duration_minutes'] - optimized['estimated_duration_minutes']
        
        if time_diff > 10:
            return f"🚀 Route optimisée recommandée! Économisez {time_diff:.0f} minutes en évitant les zones congestionnées."
        elif time_diff > 5:
            return f"✅ L'itinéraire optimisé vous fait gagner {time_diff:.0f} minutes."
        elif time_diff > 0:
            return f"ℹ️ Légère amélioration possible ({time_diff:.0f} min). Les deux routes sont similaires."
        else:
            return "📍 L'itinéraire normal est actuellement optimal."
    
    def save_route_to_db(self, comparison: Dict):
        """Save route comparison to database."""
        conn = self._get_db_connection()
        cur = conn.cursor()
        
        try:
            # Insert route request
            cur.execute("""
                INSERT INTO route_requests 
                (request_id, origin_segment, destination_segment, origin_name, destination_name, status)
                VALUES (%s, %s, %s, %s, %s, 'completed')
            """, (
                comparison['request_id'],
                comparison['origin']['segment_id'],
                comparison['destination']['segment_id'],
                comparison['origin']['name'],
                comparison['destination']['name']
            ))
            
            # Insert routes
            for route_type, route in comparison['routes'].items():
                cur.execute("""
                    INSERT INTO optimized_routes 
                    (request_id, route_type, total_distance_km, estimated_duration_minutes,
                     total_congestion_score, fuel_consumption_liters, co2_emissions_kg,
                     time_saved_minutes, route_geometry, waypoints)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    comparison['request_id'],
                    route_type,
                    route['total_distance_km'],
                    route['estimated_duration_minutes'],
                    route['avg_congestion_score'],
                    route['fuel_consumption_liters'],
                    route['co2_emissions_kg'],
                    route.get('time_saved_minutes', 0),
                    json.dumps(route['route_geometry']),
                    json.dumps(route['waypoints'])
                ))
                
                route_id = cur.fetchone()[0]
                
                # Insert segments
                for seg in route['segments']:
                    cur.execute("""
                        INSERT INTO route_segments 
                        (route_id, segment_order, segment_id, segment_name, 
                         distance_km, duration_minutes, current_speed, 
                         predicted_speed, congestion_level)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        route_id,
                        seg['segment_order'],
                        seg['segment_id'],
                        seg['segment_name'],
                        seg['distance_km'],
                        seg['duration_minutes'],
                        seg.get('current_speed'),
                        seg.get('predicted_speed'),
                        seg['congestion_level']
                    ))
            
            conn.commit()
            print(f"💾 Route comparison saved: {comparison['request_id']}")
            
        except Exception as e:
            conn.rollback()
            print(f"Error saving route: {e}")
            raise
        finally:
            cur.close()
            conn.close()


# Singleton instance
route_optimizer = RouteOptimizer()


if __name__ == "__main__":
    optimizer = RouteOptimizer()
    
    print("🗺️  Testing Route Optimizer")
    print("=" * 50)
    
    # Test route calculation
    comparison = optimizer.compare_routes('SEG001', 'SEG008')
    
    print(f"\n📍 From: {comparison['origin']['name']}")
    print(f"📍 To: {comparison['destination']['name']}")
    print(f"\n🚗 Normal Route: {comparison['routes']['normal']['estimated_duration_minutes']:.1f} min")
    print(f"🚀 Optimized Route: {comparison['routes']['optimized']['estimated_duration_minutes']:.1f} min")
    print(f"\n💡 {comparison['recommendation']}")
    print(f"\n📊 Savings: {comparison['savings']}")
