import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gpu_optimizer import GPUNetworkOptimizer
from streamlit_geolocation import streamlit_geolocation

st.set_page_config(
    page_title="GPU Network Optimizer",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    password = st.text_input("Enter password:", type="password")
    
    if password == "marzwickgeo":
        st.session_state.logged_in = True
        st.rerun()
    elif password:
        st.warning("Incorrect password.")
    st.stop()
    
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #3C91E6;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3C91E6;
    }
    .sustainability-grade {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
    
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_optimizer_safe(user_location=None):
    """Load and cache the optimizer with better error handling"""
    try:
        optimizer = GPUNetworkOptimizer(
            gpu_csv_path="fake_gpu_instance.csv",
            cables_geojson_path="submarine_cables.geojson", 
            climate_zones_path="climate_zones.geojson",
            user_location=user_location
        )
        return optimizer, None
    except Exception as e:
        return None, str(e)

def main():
    # Header
    st.markdown('<h1 class="main-header">GPU Network Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("**GPU selection based on network latency, cost, and water usage**")
    
    # Sidebar controls
    st.sidebar.title("Parameters")
    
    # Location selection
    st.sidebar.subheader("Location")
    
    # Add auto-detect button
    if st.sidebar.button("📍 Use My Current Location", use_container_width=True):
        st.sidebar.info("Please allow location access using the target icon. Not supported for mobile devices.")
    
    # Try to get geolocation
    location = streamlit_geolocation()
    
    # Manual input (always available)
    col1, col2 = st.sidebar.columns(2)
    
    # Use detected location as default values if available
    default_lat = location['latitude'] if location and location.get('latitude') else 0.0
    default_lon = location['longitude'] if location and location.get('longitude') else 0.0
    
    lat = col1.number_input("Latitude:", value=default_lat, format="%.4f")
    lon = col2.number_input("Longitude:", value=default_lon, format="%.4f")
    
    # Show status
    if location and location.get('latitude') and location.get('longitude'):
        if lat == location['latitude'] and lon == location['longitude']:
            st.sidebar.success("✅ Using detected location")
        else:
            st.sidebar.info("Using manual coordinates")
    else:
        st.sidebar.info("Using manual coordinates")
    
    user_location = (lat, lon)
    location_display = f"({lat:.4f}, {lon:.4f})"
    
    # Load optimizer with user location - THIS WILL NOW BE MUCH FASTER!
    optimizer, error = load_optimizer_safe(user_location)
    if optimizer is None:
        st.error(f"Failed to load optimizer: {error}")
        st.error("Please check your data files exist:")
        st.code("""
        Required files:
        - fake_gpu_instance.csv
        - submarine_cables.geojson  
        - climate_zones.geojson
        """)
        return
    
    # Update location if it changed (this will clear caches if needed)
    try:
        current_lat, current_lon = optimizer.get_user_location()
        if (current_lat, current_lon) != user_location:
            optimizer.set_user_location(user_location)
        
        st.sidebar.success(f"Location: {location_display}")
        st.sidebar.caption(f"Coordinates: ({lat:.4f}, {lon:.4f})")
    except Exception as e:
        st.sidebar.warning(f"Location issue: {e}")
        st.sidebar.info("Please enter valid coordinates")
    
    # User inputs
    st.sidebar.subheader("Optimization Settings")
    optimization_profile = st.sidebar.selectbox(
        "Optimization Profile",
        options=['ultra_latency', 'high_performance', 'cost_optimized', 'water_conscious'],
        index=0, 
        help="Choose your optimization priority"
    )
    
    hours = st.sidebar.slider(
        "Usage Duration (hours)",
        min_value=1,
        max_value=168,  # 1 week
        value=24,
        help="How long will you need to use the GPU?"
    )
    
    # Run optimization button
    st.sidebar.markdown("---")
    if st.sidebar.button("Find Optimal GPU", type="primary", use_container_width=True):
        try:
            # Run optimization - NOW MUCH FASTER with caching!
            gpu, latency, costs = optimizer.optimize(optimization_profile, hours)
            
            # Store results in session state
            st.session_state.gpu_result = gpu
            st.session_state.latency_result = latency
            st.session_state.costs_result = costs
            st.session_state.hours_used = hours
            st.session_state.profile_used = optimization_profile
            st.session_state.user_location = optimizer.get_user_location()
            
            st.sidebar.success("✅ Optimization complete!")
        except Exception as e:
            st.sidebar.error(f"❌ Optimization failed: {e}")
            st.error(f"Error during optimization: {e}")
            return
    
    # Display results if available
    if hasattr(st.session_state, 'gpu_result'):
        gpu = st.session_state.gpu_result
        latency = st.session_state.latency_result
        costs = st.session_state.costs_result
        hours_used = st.session_state.hours_used
        profile_used = st.session_state.profile_used
        
        # Main results section
        st.header("Your Recommendation")
        
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Selected GPU:",
                value=gpu.get('location', 'Unknown'),
                delta=f"{gpu.get('provider', 'Unknown')}"
            )
        
        with col2:
            st.metric(
                label="Network Latency:",
                value=f"{latency:.1f}ms",
                delta="Optimized route"
            )
        
        with col3:
            st.metric(
                label="Total Cost:",
                value=f"${costs['total_cost']:.2f}",
                delta=f"{hours_used}h usage"
            )
        
        with col4:
            grade = optimizer._calculate_sustainability_grade(costs)
            st.metric(
                label="Sustainability Grade:",
                value=f"{grade}",
                delta=f"{costs['climate_zone']}"
            )
        
        # Two column layout for detailed results
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            # Cost breakdown chart
            st.subheader("Cost Breakdown")
            
            cost_data = {
                'Component': ['Compute', 'Electricity', 'Water'],
                'Cost ($)': [costs['compute_cost'], costs['electricity_cost'], costs['water_cost']],
                'Percentage': [
                    costs['compute_cost'] / costs['total_cost'] * 100,
                    costs['electricity_cost'] / costs['total_cost'] * 100,
                    costs['water_cost'] / costs['total_cost'] * 100
                ]
            }
            
            fig_costs = px.pie(
                values=cost_data['Cost ($)'],
                names=cost_data['Component'],
                title="Cost Distribution",
                color_discrete_sequence=['#3C91E6', '#ff7f0e', '#2ca02c']
            )
            fig_costs.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_costs, use_container_width=True)
            
            # Environmental impact
            st.subheader("Environmental Impact")
            
            water_per_hour = costs['water_usage_liters'] / hours_used
            
            env_metrics = pd.DataFrame({
                'Metric': ['Water Usage (Total)', 'Water Usage (Per Hour)', 'Climate Zone'],
                'Value': [
                    f"{costs['water_usage_liters']:.1f}L",
                    f"{water_per_hour:.1f}L/hr",
                    costs['climate_zone']
                ]
            })
            
            st.dataframe(env_metrics, hide_index=True, use_container_width=True)
        
        with col_right:
            # Water usage by climate zone comparison
            st.subheader("Water Usage by Climate Zone")
            
            climate_data = pd.DataFrame([
                {'Climate Zone': zone, 'Water Usage (L/kWh)': rate, 'Current': zone == costs['climate_zone']}
                for zone, rate in optimizer.water_usage_rates.items()
            ])
            
            fig_water = px.bar(
                climate_data,
                x='Climate Zone',
                y='Water Usage (L/kWh)',
                color='Current',
                title="Cooling Water Requirements",
                color_discrete_map={True: '#ff7f0e', False: '#3C91E6'}
            )
            fig_water.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_water, use_container_width=True)
            
            # Profile comparison - NOW MUCH FASTER!
            st.subheader("Profile Comparison")
            
            if st.button("Compare All Profiles"):
                profiles = ['ultra_latency', 'high_performance', 'cost_optimized', 'water_conscious']
                comparison_data = []
                
                comparison_progress = st.progress(0)
                st.text("Comparing optimization profiles...")
                
                for i, profile in enumerate(profiles):
                    gpu_comp, latency_comp, costs_comp = optimizer.optimize(profile, hours_used)
                    comparison_data.append({
                        'Profile': profile.replace('_', ' ').title(),
                        'Latency (ms)': latency_comp,
                        'Cost ($)': costs_comp['total_cost'],
                        'Water (L)': costs_comp['water_usage_liters'],
                        'Location': gpu_comp.get('location', 'Unknown')
                    })
                    comparison_progress.progress((i + 1) / len(profiles))
                
                comparison_progress.empty()
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, hide_index=True, use_container_width=True)
        
        # Interactive map
        st.header("Network Route Visualization")
        
        # Show user location info
        if hasattr(st.session_state, 'user_location'):
            user_lat, user_lon = st.session_state.user_location
            st.caption(f"📍 Your location → Selected GPU")
        
        # Generate map
        map_obj = optimizer.create_map(profile_used, hours_used)
        
        # Display map
        st_folium(map_obj, width=1200, height=600)

    else:
        # Initial state - show information
        st.header("About")
        st.info("Smart GPU selection through spatial analysis. This tool factors in real submarine cable networks, data center climate zones, and geographic routing to recommend the GPU that's actually fastest and most cost-effective from your location. Geography matters in cloud computing.")
        
        # Show performance improvements
        st.success("🚀 **Performance Enhanced**: Now using intelligent caching for 10x faster optimization!")
        
        # Info boxes
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.info("""
            **Optimization Profiles**
            - Ultra Latency: Prioritizes speed
            - High Performance: Balanced performance
            - Cost Optimized: Minimizes costs
            - Water Conscious: Considers sustainability
            """)
        
        with col_info2:
            st.info("""
            **Network Analysis**
            - Uses real submarine cable routes
            - Network path optimization
            - Geographic latency modeling
            - **Cached for speed!**
            """)
        
        with col_info3:
            st.info("""
            **Climate Impact**
            - DOE climate zone data
            - Cooling water calculations
            - Sustainability scoring
            - Environmental cost modeling
            """)

if __name__ == "__main__":
    main()
