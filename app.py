import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

# Page Configuration
st.set_page_config(
    page_title="AgriSentinel | Satellite Drought & Mitigation System",
    page_icon="🌾",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"

# Session State Initialization
if "lat" not in st.session_state:
    st.session_state.lat = 28.6139  # Default to India (Delhi/Haryana region)
if "lon" not in st.session_state:
    st.session_state.lon = 77.2090
if "zoom" not in st.session_state:
    st.session_state.zoom = 6
if "search_results" not in st.session_state:
    st.session_state.search_results = []

# Main Header
st.title("🌾 AgriSentinel")
st.subheader("Field-Level Satellite Drought Monitoring & Hybrid Mitigation System")
st.markdown("---")

col_left, col_right = st.columns([1, 1], gap="large")

# --- SIDEBAR: ADMINISTRATIVE LOCATION SEARCH & CONTROLS ---
st.sidebar.header("📍 Location & Administrative Finder")

# 1. Administrative Search Bar (Country / State / District / Taluk / Village)
search_query = st.sidebar.text_input(
    "Search Location (Taluk, District, State)",
    placeholder="e.g., Nanjangud, Mysuru, Karnataka"
)

if st.sidebar.button("🔎 Search & Fly to Region", width="stretch"):
    if search_query.strip():
        with st.sidebar.spinner("Searching administrative bounds..."):
            try:
                geo_url = f"https://nominatim.openstreetmap.org/search?q={search_query}&format=json&limit=5"
                headers = {"User-Agent": "AgriSentinel-App/2.0"}
                resp = requests.get(geo_url, headers=headers, timeout=5)
                
                if resp.status_code == 200:
                    results = resp.json()
                    if results:
                        st.session_state.search_results = results
                        # Auto-center on top match
                        top_match = results[0]
                        st.session_state.lat = round(float(top_match["lat"]), 4)
                        st.session_state.lon = round(float(top_match["lon"]), 4)
                        st.session_state.zoom = 12  # Zoom level for district/taluk
                        st.sidebar.success(f"Moved map to: {top_match['display_name'].split(',')[0]}")
                        st.rerun()
                    else:
                        st.sidebar.error("No administrative region found for that query.")
            except Exception as e:
                st.sidebar.error("Location search service unavailable.")

# Dropdown if multiple administrative matches found
if st.session_state.search_results:
    match_options = {res["display_name"]: res for res in st.session_state.search_results}
    selected_name = st.sidebar.selectbox("Select Exact Region", list(match_options.keys()))
    
    if st.sidebar.button("Jump to Selected Match"):
        match_data = match_options[selected_name]
        st.session_state.lat = round(float(match_data["lat"]), 4)
        st.session_state.lon = round(float(match_data["lon"]), 4)
        st.session_state.zoom = 13
        st.rerun()

st.sidebar.markdown("---")

# 2. Manual Coordinate Controls
st.sidebar.markdown("##### Manual Coordinates")
manual_lat = st.sidebar.number_input("Latitude", value=float(st.session_state.lat), format="%.4f")
manual_lon = st.sidebar.number_input("Longitude", value=float(st.session_state.lon), format="%.4f")

if manual_lat != st.session_state.lat or manual_lon != st.session_state.lon:
    st.session_state.lat = manual_lat
    st.session_state.lon = manual_lon
    st.session_state.zoom = 13
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🌱 Crop & Field Parameters")

crop_type = st.sidebar.selectbox(
    "Select Crop Type",
    ["Wheat", "Rice / Paddy", "Maize / Corn", "Cotton", "Sugarcane", "Soybean", "Vegetables", "Pulses"]
)

growth_stage = st.sidebar.selectbox(
    "Growth Stage",
    ["Sowing / Germination", "Vegetative Growth", "Flowering / Reproductive", "Grain Filling / Podding", "Maturity / Harvest"]
)

irrigation_type = st.sidebar.selectbox(
    "Irrigation Method",
    ["Drip", "Sprinkler", "Flood / Furrow", "Rainfed"]
)

language = st.sidebar.selectbox(
    "Advisory Language",
    ["English", "Hindi", "Kannada", "Telugu", "Spanish"]
)

st.sidebar.markdown("---")
analyze_btn = st.sidebar.button("🔍 Analyze Crop Risk & Mitigation", width="stretch")


# --- LEFT COLUMN: SATELLITE MAP ---
with col_left:
    st.markdown("### 🛰️ Interactive Satellite Map")
    st.caption("Search an administrative region on the left sidebar, or click anywhere on the field.")

    m = folium.Map(
        location=[st.session_state.lat, st.session_state.lon],
        zoom_start=st.session_state.zoom,
        tiles=None,
        max_zoom=18
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite View",
        max_zoom=18
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Street Map",
        max_zoom=18
    ).add_to(m)

    folium.LayerControl().add_to(m)

    folium.Marker(
        [st.session_state.lat, st.session_state.lon],
        popup=f"Selected Pin: {st.session_state.lat:.4f}, {st.session_state.lon:.4f}",
        icon=folium.Icon(color="red", icon="leaf")
    ).add_to(m)

    map_data = st_folium(
        m,
        width="100%",
        height=480,
        key="sat_farm_map",
        returned_objects=["last_clicked", "zoom"]
    )

    if map_data:
        if map_data.get("zoom") is not None:
            st.session_state.zoom = map_data["zoom"]

        if map_data.get("last_clicked"):
            c_lat = round(map_data["last_clicked"]["lat"], 4)
            c_lon = round(map_data["last_clicked"]["lng"], 4)
            
            if c_lat != st.session_state.lat or c_lon != st.session_state.lon:
                st.session_state.lat = c_lat
                st.session_state.lon = c_lon
                st.rerun()


# --- RIGHT COLUMN: PREDICTION & MITIGATION PANEL ---
with col_right:
    st.markdown("### 📊 Crop Risk & Mitigation Advisory")
    st.info(f"**Selected Pin:** `Lat: {st.session_state.lat:.4f}`, `Lon: {st.session_state.lon:.4f}`")

    if analyze_btn:
        with st.spinner("Running telemetry inference & hybrid agronomic engine..."):
            try:
                payload = {
                    "latitude": st.session_state.lat,
                    "longitude": st.session_state.lon,
                    "crop_type": crop_type,
                    "growth_stage": growth_stage,
                    "irrigation_type": irrigation_type,
                    "language": language
                }
                response = requests.post(API_URL, json=payload, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    loc = data["location_info"]
                    pred = data["prediction"]
                    metrics = data["key_metrics_used"]
                    mitigation = data["mitigation_engine"]
                    rule_analysis = mitigation["rule_analysis"]

                    st.success(f"📍 **Location:** {loc['place_name']}")

                    risk_level = pred["risk_level"]
                    confidence = pred["confidence"] * 100

                    # Status Banner
                    if pred["class_code"] == 0:
                        st.success(f"### Risk Status: {risk_level} ({confidence:.1f}% Confidence)")
                    elif pred["class_code"] == 1:
                        st.warning(f"### Risk Status: {risk_level} ({confidence:.1f}% Confidence)")
                    else:
                        st.error(f"### Risk Status: {risk_level} ({confidence:.1f}% Confidence)")

                    # Yield Risk & Deficit Summary
                    c1, c2 = st.columns(2)
                    c1.metric("Est. Yield Loss Risk", rule_analysis["estimated_yield_risk"])
                    c2.metric("Est. Water Deficit", f"{rule_analysis['estimated_water_deficit_mm']} mm")

                    # Agronomic Advisory Section
                    st.markdown("---")
                    st.markdown("### 💡 AI & Agronomic Action Plan")
                    st.markdown(mitigation["ai_advisory"])

                    # Telemetry Metrics Grid
                    st.markdown("---")
                    st.markdown("### 🛰️ Live Field Telemetry")
                    m1, m2 = st.columns(2)
                    m1.metric("NDWI (Moisture Index)", f"{metrics['ndwi_moisture_index']:.4f}")
                    m2.metric("NDVI (Vegetation Index)", f"{metrics['ndvi_vegetation_index']:.4f}")

                    m3, m4 = st.columns(2)
                    m3.metric("7-Day Max Temp", f"{metrics['temp_7d_max_c']} °C")
                    m4.metric("7-Day Total Rain", f"{metrics['rain_7d_sum_mm']} mm")

                    # Environmental Bar Chart
                    chart_df = pd.DataFrame({
                        "Metric": ["Max Temp (°C)", "Rain Sum (mm)", "NDVI (x100)", "NDWI (x100)"],
                        "Value": [
                            metrics['temp_7d_max_c'],
                            metrics['rain_7d_sum_mm'],
                            metrics['ndvi_vegetation_index'] * 100,
                            metrics['ndwi_moisture_index'] * 100
                        ]
                    })

                    fig_bar = go.Figure(go.Bar(
                        x=chart_df["Metric"],
                        y=chart_df["Value"],
                        marker_color=['#FF5722', '#2196F3', '#4CAF50', '#00BCD4']
                    ))
                    fig_bar.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10))
                    st.plotly_chart(fig_bar, width="stretch")

                else:
                    err_msg = response.json().get("detail", "Error processing request.")
                    st.error(f"❌ **Validation Error:** {err_msg}")

            except requests.exceptions.ConnectionError:
                st.error("🔌 Could not connect to FastAPI server! Make sure Uvicorn is running on http://127.0.0.1:8000.")
    else:
        st.write("👈 Search for a location or pick your crop parameters in the sidebar, then click **Analyze Crop Risk & Mitigation**.")