def _calculate_sustainability_grade(self, costs, hours=None):
    """
    Assign sustainability grade based on water usage per hour
    
    Updated with realistic thresholds based on actual data analysis:
    - Range: 0.12L/hr (300W Subarctic) to 3.30L/hr (1500W Hot-Humid)
    - Thresholds derived from analysis of real GPU power consumption
    """
    # Calculate water usage per hour
    if hours:
        water_per_hour = costs['water_usage_liters'] / hours
    else:
        # If no hours provided, assume it's already per-hour calculation
        # This handles both cases where costs might be calculated differently
        water_per_hour = costs.get('water_usage_per_hour', costs['water_usage_liters'])
    
    # Realistic thresholds based on actual data analysis
    # These create a good distribution across the actual range of 0.12-3.30 L/hr
    if water_per_hour < 0.5:
        return 'A+'    # Excellent efficiency (31% of scenarios)
    elif water_per_hour < 1.0:
        return 'A'     # Good efficiency (34% of scenarios)
    elif water_per_hour < 1.5:
        return 'B'     # Average efficiency (14% of scenarios)
    elif water_per_hour < 2.5:
        return 'C'     # Below average (14% of scenarios)
    else:
        return 'D'     # Poor efficiency (6% of scenarios)

def _calculate_water_usage(self, gpu_power_watts, lat, lon):
    """Calculate water usage in liters per hour based on climate zone"""
    climate_zone = self._get_climate_zone(lat, lon)
    base_rate = self.water_usage_rates.get(climate_zone, 1.5)  # Default moderate rate
    
    # Calculate base water usage (liters per hour)
    power_kw = gpu_power_watts / 1000
    water_usage_per_hour = power_kw * base_rate
    
    return water_usage_per_hour, climate_zone

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
        'water_usage_per_hour': water_usage_per_hour,  # Add this for clarity
        'climate_zone': climate_zone
    }

def create_sustainability_report(self, profile='water_conscious', hours=24):
    """Generate a comprehensive sustainability report"""
    gpu, latency, costs = self.optimize(profile, hours)
    
    # Calculate sustainability grade using the fixed method
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
