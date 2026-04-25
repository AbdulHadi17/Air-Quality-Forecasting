import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

st.set_page_config(page_title="Lahore AQI Forecast", layout="wide", page_icon="🌤️")

# Custom CSS for premium aesthetic
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1, h2, h3 {
        color: #00E676;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌍 Lahore Air Quality (PM2.5) Spatio-Temporal Forecast")
st.markdown("Real-time prediction grid powered by Hybrid ConvLSTM.")

API_URL = "https://abdulhadi17-aqi-backend.hf.space/predict/recent"

@st.cache_data(ttl=60)
def fetch_prediction():
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return None

with st.sidebar:
    st.header("Control Panel")
    if st.button("Refresh Latest Forecast"):
        st.cache_data.clear()
    
    st.markdown("---")
    st.markdown("**Model:** ConvLSTM Spatial Grid")
    st.markdown("**Resolution:** ~1km (0.005°)")
    st.markdown("**City:** Lahore, Pakistan")

data = fetch_prediction()

if data:
    lats = data["grid_lats"]
    lons = data["grid_lons"]
    preds = data["predictions"]
    
    heat_data = []
    max_val = 0
    avg_val = 0
    valid_points = 0
    
    for i in range(len(lats)):
        for j in range(len(lons)):
            val = preds[i][j]
            if val > 0: 
                heat_data.append([lats[i], lons[j], val])
                if val > max_val:
                    max_val = val
                avg_val += val
                valid_points += 1
                
    if valid_points > 0:
        avg_val = avg_val / valid_points

    col1, col2 = st.columns([3, 1])
    
    with col1:
        center_lat = sum(lats)/len(lats) if lats else 31.4895
        center_lon = sum(lons)/len(lons) if lons else 74.3115
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB dark_matter")
        
        HeatMap(
            heat_data,
            name="PM2.5 Forecast",
            radius=15,
            blur=10,
            max_val=max_val,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
        ).add_to(m)
        
        st_folium(m, width=900, height=600, returned_objects=[])

    with col2:
        st.subheader("Grid Statistics")
        st.metric(label="Peak PM2.5", value=f"{max_val:.1f} µg/m³")
        st.metric(label="Average PM2.5", value=f"{avg_val:.1f} µg/m³")
        st.metric(label="Grid Cells", value=valid_points)
        
        def get_aqi_color(val):
            if val <= 12.0: return "🟢 Good"
            elif val <= 35.4: return "🟡 Moderate"
            elif val <= 55.4: return "🟠 Unhealthy for Sensitive"
            elif val <= 150.4: return "🔴 Unhealthy"
            elif val <= 250.4: return "🟣 Very Unhealthy"
            else: return "🟤 Hazardous"
            
        st.markdown(f"**Overall Status:** {get_aqi_color(avg_val)}")
else:
    st.error("Waiting for FastAPI backend to be available at localhost:8000... Or no mock data is available.")
