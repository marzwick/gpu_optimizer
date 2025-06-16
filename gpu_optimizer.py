import pandas as pd
import geocoder
import math
import geopandas as gpd
import json
import re
from shapely.geometry import Point
from geopy.distance import geodesic
import networkx as nx
import folium

class GPUNetworkOptimizer:
    def __init__(self, gpu_csv_path, cables_geojson_path, climate_zones_path, user_location=None):
        self.gpu_data = pd.read_csv(gpu_csv_path)
        
        # Set user location - either provided coordinates or auto-detect
        if user_location is not None:
            self.user_lat, self.user_lon = user_location
        else:
            try:
                self.user_lat, self.user_lon = geocoder.ip('me').latlng
            except:
                # Fallback to default location if geocoding fails
                self.user_lat, self.user_lon = 37.7749, -122.4194  # San Francisco
        
        self.cables_gdf = gpd.read_file(cables_geojson_path)
        self.climate_zones_gdf = gpd.read_file(climate_zones_path)
        self.landing_points_gdf = self._build_landing_points()
        self.cable_network = self._build_network()
        
        # Water usage multipliers by climate zone (liters per kWh)
        self.water_usage_rates = {
            'Hot-Humid': 2.2,      # Zones 1-2A: High cooling demand, humid
            'Hot-Dry': 1.8,        # Zones 2B-3B: High cooling, dry air helps
            'Mixed-Humid': 1.4,    # Zones 3A-4A: Moderate cooling needs
            'Mixed-Dry': 1.1,      # Zones 3B-4B: Lower humidity helps
            'Cold': 0.8,           # Zones 5-6: Low cooling needs
            'Very Cold': 0.6,      # Zones 7-8: Minimal cooling
            'Subarctic': 0.4       # Zone 8+: Natural cooling available
        }
        
        # Pre-calculate dataset statistics for normalization
        self._calculate_dataset_stats()
    
    def _calculate_dataset_stats(self):
        """Calculate dataset statistics for proper normalization"""
        self.stats = {
            'min_cost': self.gpu_data['hourly_cost_usd'].min(),
            'max_cost': self.gpu_data['hourly_cost_usd'].max(),
            'min_perf': self.gpu_data['compute_capability'].min(),
            'max_perf': self.gpu_data['compute_capability'].max(),
            'min_power': self.gpu_data['gpu_power_watts'].min(),
            'max_power': self.gpu_data['gpu_power_watts'].max(),
            'min_elec': self.gpu_data['electricity_price_kwh'].min(),
            'max_elec': self.gpu_data['electricity_price_kwh'].max()
        }
        
        print("=== Dataset Statistics ===")
        print(f"Cost range: ${self.stats['min_cost']:.2f} - ${self.stats['max_cost']:.2f}")
        print(f"Performance range: {self.stats['min_perf']:.1f} - {self.stats['max_perf']:.1f}")
        print(f"Power range: {self.stats['min_power']:.0f}W - {self.stats['max_power']:.0f}W")
        print(f"Electricity range: ${self.stats['min_elec']:.2f} - ${self.stats['max_elec']:.2f}/kWh")
    
    def set_user_location(self, location_input):
        """
        Set user location from coordinates or auto-detection
        
        Args:
            location_input: Can be:
                - Tuple of (lat, lon) coordinates
                - 'auto' to use IP-based detection
        """
        if location_input == 'auto':
            try:
                self.user_lat, self.user_lon = geocoder.ip('me').latlng
            except:
                # Fallback to default location
                self.user_lat, self.user_lon = 37.7749, -122.4194
        elif isinstance(location_input, (tuple, list)) and len(location_input) == 2:
            self.user_lat, self.user_lon = location_input
        else:
            raise ValueError("Location must be 'auto' or (lat, lon) tuple")
    
    def get_user_location(self):
        """Return current user location as (lat, lon) tuple"""
        return (self.user_lat, self.user_lon)
    
    def _get_climate_zone(self, lat, lon):
        """Determine climate zone for a given location using the GeoJSON data"""
        point = Point(lon, lat)
        
        # Check which climate zone polygon contains this point
        for _, zone in self.climate_zones_gdf.iterrows():
            if zone.geometry.contains(point):
                return zone['BA_Climate_Zone']
        
        # Fallback if point not in any zone (maybe outside North America)
        return self._estimate_climate_zone(lat)
    
    def _estimate_climate_zone(self, lat):
        """Fallback climate estimation based on latitude"""
        if lat > 60: return 'Subarctic'
        elif lat > 50: return 'Very Cold'
        elif lat > 42: return 'Cold'
        elif lat > 35: return 'Mixed-Humid'
        elif lat > 28: return 'Hot-Dry'
        else: return 'Hot-Humid'
    
    def _calculate_water_usage(self, gpu_power_watts, lat, lon):
        """Calculate water usage in liters per hour based on climate zone"""
        climate_zone = self._get_climate_zone(lat, lon)
        base_rate = self.water_usage_rates.get(climate_zone, 1.5)  # Default moderate rate
        
        # Calculate base water usage (liters per hour)
        power_kw = gpu_power_watts / 1000
        water_usage_per_hour = power_kw * base_rate
        
        return water_usage_per_hour, climate_zone
    
    def _calculate_water_cost(self, water_usage_liters_per_hour, hours):
        """Calculate water cost using average pricing"""
        # Simple average water cost in USD per 1000 liters
        cost_per_1000l = 3.00  # National average
        
        total_liters = water_usage_liters_per_hour * hours
        return (total_liters / 1000) * cost_per_1000l
    
    def _build_landing_points(self):
        all_points = {}
        for _, cable in self.cables_gdf.iterrows():
            points = self._extract_landing_points(cable['description'])
            for lp in points:
                lp_id = lp['landing_point_id']
                if lp_id not in all_points:
                    lat, lon = map(float, lp['latlon'].split(','))
                    all_points[lp_id] = {'id': lp_id, 'name': lp['name'], 'latitude': lat, 'longitude': lon}
        
        df = pd.DataFrame(all_points.values())
        return gpd.GeoDataFrame(df, geometry=[Point(row['longitude'], row['latitude']) for _, row in df.iterrows()], crs='EPSG:4326')
    
    def _extract_landing_points(self, description):
        match = re.search(r'\[{.*?}\]', description, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0).replace('\\"', '"'))
            except:
                return []
        return []
    
    def _build_network(self):
        G = nx.Graph()
        for _, lp in self.landing_points_gdf.iterrows():
            G.add_node(lp['id'], lat=lp['latitude'], lon=lp['longitude'], name=lp['name'])
        
        for _, cable in self.cables_gdf.iterrows():
            if cable['name'] == 'Arctic Fibre':
                continue
            points = self._extract_landing_points(cable['description'])
            if len(points) >= 2:
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        id1, id2 = points[i]['landing_point_id'], points[j]['landing_point_id']
                        if id1 in G.nodes and id2 in G.nodes:
                            coords1 = list(map(float, points[i]['latlon'].split(',')))
                            coords2 = list(map(float, points[j]['latlon'].split(',')))
                            distance = geodesic(coords1, coords2).kilometers
                            G.add_edge(id1, id2, weight=distance, cable_name=cable['name'])
        
        self._add_terrestrial_connections(G)
        return G
    
    def _add_terrestrial_connections(self, G):
        west_coast = self.landing_points_gdf[
            (self.landing_points_gdf['name'].str.contains('California|Oregon|Washington', na=False)) &
            (self.landing_points_gdf['longitude'] < -110)
        ]
        east_coast = self.landing_points_gdf[
            (self.landing_points_gdf['name'].str.contains('New York|New Jersey|Massachusetts|Virginia', na=False)) &
            (self.landing_points_gdf['longitude'] > -80)
        ]
        
        for _, west in west_coast.iterrows():
            for _, east in east_coast.iterrows():
                G.add_edge(west['id'], east['id'], weight=4500, cable_name='US_Terrestrial')
    
    def _haversine_latency(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat1 - lat2, lon1 - lon2
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return (6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)) * 0.02) + 5
    
    def _network_latency(self, gpu_lat, gpu_lon):
        user_nearest = self._find_nearest_points(self.user_lat, self.user_lon, 3)
        gpu_nearest = self._find_nearest_points(gpu_lat, gpu_lon, 3)
        
        min_distance = float('inf')
        for _, u_lp in user_nearest.iterrows():
            for _, g_lp in gpu_nearest.iterrows():
                try:
                    path_length = nx.shortest_path_length(self.cable_network, u_lp['id'], g_lp['id'], weight='weight')
                    user_to_lp = geodesic((self.user_lat, self.user_lon), (u_lp['latitude'], u_lp['longitude'])).kilometers
                    gpu_to_lp = geodesic((gpu_lat, gpu_lon), (g_lp['latitude'], g_lp['longitude'])).kilometers
                    total = user_to_lp + path_length + gpu_to_lp
                    min_distance = min(min_distance, total)
                except nx.NetworkXNoPath:
                    continue
        
        return (min_distance * 0.02) + 5 if min_distance != float('inf') else self._haversine_latency(self.user_lat, self.user_lon, gpu_lat, gpu_lon)
    
    def _find_nearest_points(self, lat, lon, n):
        distances = self.landing_points_gdf.geometry.apply(lambda lp: geodesic((lat, lon), (lp.y, lp.x)).kilometers)
        return self.landing_points_gdf.loc[distances.nsmallest(n).index]
    
    def _calculate_total_cost(self, row, hours):
        """Enhanced cost calculation including water costs"""
        # Base compute cost
        compute_cost = row['hourly_cost_usd'] * hours
        
        # Electricity cost
        electricity_cost = (row['gpu_power_watts'] / 1000) * row['electricity_price_kwh'] * hours
        
        # Water usage and cost
        water_usage_per_hour, climate_zone = self._calculate_water_usage(
            row['gpu_power_watts'], 
            row['latitude'], 
            row['longitude']
        )
        
        total_water_usage = water_usage_per_hour * hours
        water_cost = self._calculate_water_cost(water_usage_per_hour, hours)
        
        return {
            'total_cost': compute_cost + electricity_cost + water_cost,
            'compute_cost': compute_cost,
            'electricity_cost': electricity_cost,
            'water_cost': water_cost,
            'water_usage_liters': total_water_usage,
            'water_usage_per_hour': water_usage_per_hour,  # FIXED: Added this line
            'climate_zone': climate_zone
        }
    
    def _normalize_score(self, value, min_val, max_val, invert=False):
        """Normalize a value to 0-1 range, optionally inverted"""
        if max_val == min_val:
            return 0.5  # If no variation, return neutral score
        
        normalized = (value - min_val) / (max_val - min_val)
        return (1 - normalized) if invert else normalized
    
    def optimize(self, profile, hours):
        """Enhanced optimization with better scoring and normalization"""
        # Balanced weight profiles that prioritize different aspects accurately
        weight_profiles = {
            'ultra_latency': {'latency': 0.7, 'performance': 0.2, 'cost': 0.1, 'water_impact': 0.0},
            'high_performance': {'latency': 0.2, 'performance': 0.6, 'cost': 0.2, 'water_impact': 0.0},
            'cost_optimized': {'latency': 0.1, 'performance': 0.2, 'cost': 0.7, 'water_impact': 0.0},
            'water_conscious': {'latency': 0.2, 'performance': 0.2, 'cost': 0.3, 'water_impact': 0.3}
        }
        
        weights = weight_profiles[profile]
        
        # Pre-calculate all metrics for proper normalization
        all_latencies = []
        all_costs = []
        all_water_usage = []
        all_performance = []
        all_rows = []
        
        for _, row in self.gpu_data.iterrows():
            latency = self._network_latency(row['latitude'], row['longitude'])
            costs = self._calculate_total_cost(row, hours)
            
            all_latencies.append(latency)
            all_costs.append(costs['total_cost'])
            all_water_usage.append(costs['water_usage_liters'])
            all_performance.append(row['compute_capability'])
            all_rows.append((row, latency, costs))
        
        # Get min/max for normalization
        min_latency, max_latency = min(all_latencies), max(all_latencies)
        min_cost, max_cost = min(all_costs), max(all_costs)
        min_water, max_water = min(all_water_usage), max(all_water_usage)
        min_perf, max_perf = min(all_performance), max(all_performance)
        
        print(f"\n=== {profile.upper()} PROFILE ANALYSIS ===")
        print(f"Latency range: {min_latency:.1f} - {max_latency:.1f}ms")
        print(f"Cost range: ${min_cost:.2f} - ${max_cost:.2f}")
        print(f"Water range: {min_water:.1f} - {max_water:.1f}L")
        print(f"Performance range: {min_perf:.1f} - {max_perf:.1f}")
        
        best_score, best_gpu, best_latency, best_costs = float('inf'), None, 0, {}
        all_scores = []
        
        for row, latency, costs in all_rows:
            # Normalize all metrics (0-1 scale)
            norm_latency = self._normalize_score(latency, min_latency, max_latency)
            norm_cost = self._normalize_score(costs['total_cost'], min_cost, max_cost)
            norm_water = self._normalize_score(costs['water_usage_liters'], min_water, max_water)
            norm_performance = self._normalize_score(row['compute_capability'], min_perf, max_perf, invert=True)  # Higher is better
            
            # Calculate weighted score (lower is better for all components now)
            score = (weights['latency'] * norm_latency + 
                    weights['cost'] * norm_cost + 
                    weights['water_impact'] * norm_water + 
                    weights['performance'] * norm_performance)
            
            all_scores.append({
                'location': row.get('location', 'Unknown'),
                'provider': row.get('provider', 'Unknown'),
                'instance': row.get('instance_type', 'Unknown'),
                'latency': latency,
                'cost': costs['total_cost'],
                'water': costs['water_usage_liters'],
                'performance': row['compute_capability'],
                'norm_latency': norm_latency,
                'norm_cost': norm_cost,
                'norm_water': norm_water,
                'norm_performance': norm_performance,
                'final_score': score
            })
            
            if score < best_score:
                best_score, best_gpu, best_latency, best_costs = score, row, latency, costs
        
        # Debug: Print top 5 GPUs for this profile
        sorted_scores = sorted(all_scores, key=lambda x: x['final_score'])
        print("Top 5 GPUs:")
        for i, gpu_score in enumerate(sorted_scores[:5]):
            print(f"  {i+1}. {gpu_score['provider']} {gpu_score['instance']} in {gpu_score['location']}")
            print(f"     Score: {gpu_score['final_score']:.4f} | Latency: {gpu_score['latency']:.1f}ms | "
                  f"Cost: ${gpu_score['cost']:.2f} | Water: {gpu_score['water']:.1f}L | Perf: {gpu_score['performance']:.1f}")
            print(f"     Normalized - Lat: {gpu_score['norm_latency']:.3f}, Cost: {gpu_score['norm_cost']:.3f}, "
                  f"Water: {gpu_score['norm_water']:.3f}, Perf: {gpu_score['norm_performance']:.3f}")
        
        return best_gpu, best_latency, best_costs
    
    def create_sustainability_report(self, profile='water_conscious', hours=24):
        """Generate a comprehensive sustainability report"""
        gpu, latency, costs = self.optimize(profile, hours)
        
        # Calculate sustainability grade using the FIXED method
        sustainability_grade = self._calculate_sustainability_grade(costs, hours)
        
        report = {
            'selected_gpu': {
                'provider': gpu.get('provider', 'Unknown'),
                'location': gpu.get('location', 'Unknown'),
                'climate_zone': costs['climate_zone'],
                'gpu_power_watts': gpu['gpu_power_watts']
            },
            'performance_metrics': {
                'latency_ms': round(latency, 1),
                'compute_capability': gpu['compute_capability']
            },
            'cost_breakdown': {
                'total_cost_usd': round(costs['total_cost'], 2),
                'compute_cost_usd': round(costs['compute_cost'], 2),
                'electricity_cost_usd': round(costs['electricity_cost'], 2),
                'water_cost_usd': round(costs['water_cost'], 2)
            },
            'environmental_impact': {
                'water_usage_liters_total': round(costs['water_usage_liters'], 1),
                'water_usage_per_hour': round(costs['water_usage_per_hour'], 2),
                'climate_zone': costs['climate_zone'],
                'sustainability_grade': sustainability_grade,
                'grade_explanation': self._get_grade_explanation(sustainability_grade, costs['water_usage_per_hour'])
            }
        }
        
        return report
    
    def _calculate_sustainability_grade(self, costs, hours=24):
        """
        FIXED: Assign sustainability grade based on water usage per hour
        
        Updated with realistic thresholds based on actual data analysis:
        - Range: 0.12L/hr (300W Subarctic) to 3.30L/hr (1500W Hot-Humid)
        - Thresholds derived from analysis of real GPU power consumption
        """
        # Calculate water usage per hour
        if 'water_usage_per_hour' in costs:
            water_per_hour = costs['water_usage_per_hour']
        else:
            # Fallback calculation
            water_per_hour = costs['water_usage_liters'] / hours
        
        # DEBUG: Print the actual values to verify calculation
        print(f"DEBUG - Sustainability Grade Calculation:")
        print(f"  Total water usage: {costs.get('water_usage_liters', 'Unknown')} L")
        print(f"  Hours: {hours}")
        print(f"  Water per hour: {water_per_hour:.3f} L/hr")
        print(f"  Climate zone: {costs.get('climate_zone', 'Unknown')}")
        
        # FIXED THRESHOLDS - Based on actual data analysis
        if water_per_hour < 0.5:
            grade = 'A+'    # Excellent efficiency (31% of scenarios)
        elif water_per_hour < 1.0:
            grade = 'A'     # Good efficiency (34% of scenarios)
        elif water_per_hour < 1.5:
            grade = 'B'     # Average efficiency (14% of scenarios)
        elif water_per_hour < 2.5:
            grade = 'C'     # Below average (14% of scenarios)
        else:
            grade = 'D'     # Poor efficiency (6% of scenarios)
        
        print(f"  Final grade: {grade}")
        return grade
    
    def _get_grade_explanation(self, grade, water_per_hour):
        """Provide explanation for the sustainability grade"""
        explanations = {
            'A+': f"Excellent water efficiency at {water_per_hour:.2f}L/hr. Among the most sustainable options.",
            'A': f"Good water efficiency at {water_per_hour:.2f}L/hr. Above average sustainability.",
            'B': f"Average water efficiency at {water_per_hour:.2f}L/hr. Moderate environmental impact.",
            'C': f"Below average water efficiency at {water_per_hour:.2f}L/hr. Higher environmental impact.",
            'D': f"Poor water efficiency at {water_per_hour:.2f}L/hr. Significant environmental impact."
        }
        return explanations.get(grade, f"Water usage: {water_per_hour:.2f}L/hr")
    
    def create_map(self, profile='ultra_latency', hours=24, show_climate_zones=False):
        """Enhanced map with climate zone and water usage visualization"""
        gpu, latency, costs = self.optimize(profile, hours)
        m = folium.Map(location=[self.user_lat, self.user_lon], zoom_start=3)
        
        # User location
        folium.Marker([self.user_lat, self.user_lon], popup="Your Location", 
                     icon=folium.Icon(color='blue', icon='user', prefix='fa')).add_to(m)
        
        # Climate zones layer - simplified and visible by default if requested
        if show_climate_zones:
            try:
                # Simplify geometries to improve performance
                simplified_zones = self.climate_zones_gdf.copy()
                simplified_zones['geometry'] = simplified_zones.geometry.simplify(0.01, preserve_topology=True)
                
                # Add zones with better styling
                for _, zone in simplified_zones.iterrows():
                    # Get zone name safely
                    zone_name = zone.get('BA_Climate_Zone', 'Unknown Zone')
                    
                    folium.GeoJson(
                        zone.geometry,
                        style_function=lambda x, zone_name=zone_name: {
                            'fillColor': self._get_zone_color(zone_name),
                            'color': 'darkblue',
                            'weight': 2,
                            'fillOpacity': 0.4,
                            'opacity': 0.8
                        },
                        popup=folium.Popup(f"Climate Zone: {zone_name}", parse_html=True),
                        tooltip=f"Zone: {zone_name}"
                    ).add_to(m)
                    
            except Exception as e:
                print(f"Warning: Could not add climate zones to map: {e}")
        
        # Network route
        user_nearest = self._find_nearest_points(self.user_lat, self.user_lon, 1).iloc[0]
        gpu_nearest = self._find_nearest_points(gpu['latitude'], gpu['longitude'], 1).iloc[0]
        
        try:
            path = nx.shortest_path(self.cable_network, user_nearest['id'], gpu_nearest['id'], weight='weight')
            coords = [[self.user_lat, self.user_lon]]
            for point_id in path:
                node = self.cable_network.nodes[point_id]
                coords.append([node['lat'], node['lon']])
            coords.append([gpu['latitude'], gpu['longitude']])
            
            folium.PolyLine(coords, color='red', weight=3, popup=f"Route: {latency:.1f}ms").add_to(m)
        except:
            folium.PolyLine([[self.user_lat, self.user_lon], [gpu['latitude'], gpu['longitude']]], 
                          color='red', weight=3, dash_array='10,10').add_to(m)
        
        # Enhanced GPU marker with water usage info
        sustainability_grade = self._calculate_sustainability_grade(costs, hours)
        popup_text = f"""
        <b>{gpu.get('provider', 'Unknown')} {gpu.get('instance_type', '')}</b><br>
        Location: {gpu.get('location', 'Unknown')}<br>
        Climate Zone: {costs['climate_zone']}<br>
        Latency: {latency:.1f}ms<br>
        Total Cost: ${costs['total_cost']:.2f}<br>
        Water Usage: {costs['water_usage_per_hour']:.2f}L/hr<br>
        Sustainability: {sustainability_grade}
        """
        
        # Color code by sustainability grade
        marker_color = {'A+': 'green', 'A': 'lightgreen', 'B': 'orange', 'C': 'red', 'D': 'darkred'}[sustainability_grade]
        
        folium.Marker([gpu['latitude'], gpu['longitude']], popup=popup_text,
                     icon=folium.Icon(color=marker_color, icon='microchip', prefix='fa')).add_to(m)
        
        return m
    
    def _get_zone_color(self, zone_name):
        """Get color for climate zone"""
        zone_colors = {
            'Hot-Humid': '#ff4444',
            'Hot-Dry': '#ff8844', 
            'Mixed-Humid': '#ffaa44',
            'Mixed-Dry': '#ffdd44',
            'Cold': '#44aaff',
            'Very Cold': '#4488ff',
            'Subarctic': '#4444ff'
        }
        return zone_colors.get(zone_name, '#888888')
