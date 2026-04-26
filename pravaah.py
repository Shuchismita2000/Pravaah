"""
PRAVAAH — Renewable Energy Intelligence Dashboard
Run with: streamlit run pravaah_app.py
Dependencies: streamlit pandas plotly requests numpy
"""

import streamlit as st

st.set_page_config(
    page_title="Pravaah",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Sidebar branding */
  [data-testid="stSidebar"] { background: #0f1117; }
  [data-testid="stSidebar"] * { color: #e8eaf0 !important; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 12px 18px;
  }

  /* Section headers */
  .section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #7eb8f7;
    border-left: 3px solid #7eb8f7;
    padding-left: 8px;
    margin: 18px 0 10px 0;
  }

  /* Weather card */
  .weather-card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Pravaah")
    st.markdown("*Renewable Energy Intelligence*")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏭 Plant Operations", "🌦️ Weather Report"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Data: Karnataka RE Portfolio · 50 plants")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — PLANT OPERATIONS
# ════════════════════════════════════════════════════════════════════════════
if page == "🏭 Plant Operations":

    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np

    st.title("🏭 Plant Operations Dashboard")
    st.caption("Karnataka Renewable Energy Portfolio · Live data view")

    # ── Load data ────────────────────────────────────────────────────────────
    @st.cache_data(show_spinner="Loading plant master…")
    def load_plant_master():
        return pd.read_csv("data/plant_master.csv")

    @st.cache_data(show_spinner="Loading lifecycle events…")
    def load_lifecycle():
        df = pd.read_csv("data/lifecycle_events.csv")
        df["event_month"] = pd.to_datetime(df["event_month"])
        return df

    @st.cache_data(show_spinner="Loading generation data (this may take a moment)…")
    def load_generation():
        df = pd.read_csv(
            "data/generation.csv",
            parse_dates=["timestamp"],
            dtype={"plant_id": "category", "plant_type": "category",
                   "region": "category", "status": "category"},
        )
        return df

    plants = load_plant_master()
    lifecycle = load_lifecycle()

    with st.spinner("Loading 2.8M generation records…"):
        gen = load_generation()

    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Filters")
        plant_type_opts = ["All"] + sorted(plants["plant_type"].unique().tolist())
        sel_type = st.selectbox("Plant type", plant_type_opts)

        region_opts = ["All"] + sorted(plants["region"].unique().tolist())
        sel_region = st.selectbox("Region", region_opts)

        date_min = gen["timestamp"].min().date()
        date_max = gen["timestamp"].max().date()
        sel_dates = st.date_input(
            "Date range",
            value=(date_max - pd.Timedelta(days=30), date_max),
            min_value=date_min,
            max_value=date_max,
        )
        if isinstance(sel_dates, (list, tuple)) and len(sel_dates) == 2:
            d_start, d_end = sel_dates
        else:
            d_start = d_end = sel_dates if not isinstance(sel_dates, (list, tuple)) else sel_dates[0]

    # Filter plants
    fp = plants.copy()
    if sel_type != "All":
        fp = fp[fp["plant_type"] == sel_type]
    if sel_region != "All":
        fp = fp[fp["region"] == sel_region]

    plant_ids = fp["plant_id"].tolist()

    # Filter generation
    fg = gen[
        (gen["plant_id"].isin(plant_ids)) &
        (gen["timestamp"].dt.date >= d_start) &
        (gen["timestamp"].dt.date <= d_end)
    ]

    # ── KPI Row ───────────────────────────────────────────────────────────────
    total_cap = fp["capacity_mw"].sum()
    total_gen = fg["actual_generation_mw"].sum() / 1000  # GWh (hourly MW → MWh → GWh)
    avg_health = fg["health_factor"].mean() if len(fg) else 0
    curtailed = fg["curtailment_mw"].sum() / 1000
    n_on = fg[fg["status"] == "ON"]["plant_id"].nunique()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Plants", f"{len(fp)}", f"{sel_type if sel_type != 'All' else 'All types'}")
    k2.metric("Total Capacity", f"{total_cap:,.0f} MW")
    k3.metric("Generation (period)", f"{total_gen:,.1f} GWh")
    k4.metric("Avg Health Factor", f"{avg_health:.3f}")
    k5.metric("Curtailment", f"{curtailed:,.1f} GWh")

    st.divider()

    # ── Row 1: Generation trend + Plant type split ────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">Generation Trend (Daily)</div>', unsafe_allow_html=True)
        daily = (
            fg.groupby([fg["timestamp"].dt.date, "plant_type"])["actual_generation_mw"]
            .sum()
            .reset_index()
        )
        daily.columns = ["date", "plant_type", "generation_mwh"]
        daily["generation_gwh"] = daily["generation_mwh"] / 1000
        fig_trend = px.area(
            daily, x="date", y="generation_gwh", color="plant_type",
            labels={"generation_gwh": "Generation (GWh)", "date": ""},
            color_discrete_map={"Solar": "#f5a623", "Wind": "#4fc3f7", "Hybrid": "#a8d8a8"},
            template="plotly_dark",
        )
        fig_trend.update_layout(
            legend_title="", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Capacity by Type</div>', unsafe_allow_html=True)
        cap_split = fp.groupby("plant_type")["capacity_mw"].sum().reset_index()
        fig_pie = px.pie(
            cap_split, names="plant_type", values="capacity_mw",
            color="plant_type",
            color_discrete_map={"Solar": "#f5a623", "Wind": "#4fc3f7", "Hybrid": "#a8d8a8"},
            template="plotly_dark", hole=0.55,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=280,
            margin=dict(t=10, b=10), legend_title="",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Row 2: Health heatmap + Curtailment bar ───────────────────────────────
    col_a, col_b = st.columns([2, 3])

    with col_a:
        st.markdown('<div class="section-header">Plant Health Snapshot</div>', unsafe_allow_html=True)
        health_snap = (
            fg.groupby("plant_id")["health_factor"]
            .mean()
            .reset_index()
            .merge(fp[["plant_id", "plant_name", "plant_type"]], on="plant_id")
            .sort_values("health_factor")
        )
        health_snap["color"] = health_snap["health_factor"].apply(
            lambda x: "#e74c3c" if x < 0.7 else ("#f39c12" if x < 0.85 else "#2ecc71")
        )
        fig_health = go.Figure(go.Bar(
            x=health_snap["health_factor"],
            y=health_snap["plant_name"],
            orientation="h",
            marker_color=health_snap["color"],
            text=health_snap["health_factor"].apply(lambda x: f"{x:.3f}"),
            textposition="outside",
        ))
        fig_health.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=340,
            margin=dict(t=5, b=5, l=5, r=60),
            xaxis=dict(range=[0, 1.05]),
            yaxis=dict(tickfont=dict(size=9)),
        )
        st.plotly_chart(fig_health, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Top 10 Plants by Generation</div>', unsafe_allow_html=True)
        top10 = (
            fg.groupby("plant_id")["actual_generation_mw"]
            .sum()
            .div(1000)  # GWh
            .reset_index()
            .merge(fp[["plant_id", "plant_name", "plant_type"]], on="plant_id")
            .nlargest(10, "actual_generation_mw")
        )
        fig_top = px.bar(
            top10, x="plant_name", y="actual_generation_mw", color="plant_type",
            labels={"actual_generation_mw": "Generation (GWh)", "plant_name": ""},
            color_discrete_map={"Solar": "#f5a623", "Wind": "#4fc3f7", "Hybrid": "#a8d8a8"},
            template="plotly_dark",
        )
        fig_top.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=340, margin=dict(t=5, b=5), legend_title="",
            xaxis=dict(tickangle=-30, tickfont=dict(size=9)),
        )
        st.plotly_chart(fig_top, use_container_width=True)

    # ── Row 3: Map + Lifecycle events ────────────────────────────────────────
    col_m, col_e = st.columns([3, 2])

    with col_m:
        st.markdown('<div class="section-header">Plant Locations (Karnataka)</div>', unsafe_allow_html=True)
        map_df = fp.merge(
            fg.groupby("plant_id")["actual_generation_mw"].sum().div(1000).reset_index()
            .rename(columns={"actual_generation_mw": "gen_gwh"}),
            on="plant_id", how="left"
        ).fillna({"gen_gwh": 0})

        fig_map = px.scatter_mapbox(
            map_df, lat="latitude", lon="longitude",
            color="plant_type", size="capacity_mw",
            hover_name="plant_name",
            hover_data={"capacity_mw": True, "gen_gwh": ":.1f", "developer": True},
            color_discrete_map={"Solar": "#f5a623", "Wind": "#4fc3f7", "Hybrid": "#a8d8a8"},
            mapbox_style="carto-darkmatter",
            zoom=6, center={"lat": 15.3, "lon": 76.8},
            template="plotly_dark",
        )
        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=380,
            margin=dict(t=0, b=0, l=0, r=0), legend_title="",
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_e:
        st.markdown('<div class="section-header">Lifecycle Events</div>', unsafe_allow_html=True)
        lc_filt = lifecycle[lifecycle["plant_id"].isin(plant_ids)].copy()
        lc_filt = lc_filt.merge(fp[["plant_id", "plant_name"]], on="plant_id", how="left")
        lc_filt["event_month"] = lc_filt["event_month"].dt.strftime("%Y-%m")
        lc_filt = lc_filt.sort_values("event_month", ascending=False)

        if lc_filt.empty:
            st.info("No lifecycle events for selected filters.")
        else:
            st.dataframe(
                lc_filt[["event_month", "plant_name", "event_type", "health_after", "health_boost", "notes"]]
                .rename(columns={
                    "event_month": "Month", "plant_name": "Plant",
                    "event_type": "Event", "health_after": "Health After",
                    "health_boost": "Boost", "notes": "Notes"
                }),
                use_container_width=True,
                hide_index=True,
                height=340,
            )

    # ── Row 4: Irradiance vs Generation (Solar only) ─────────────────────────
    st.markdown('<div class="section-header">Irradiance vs Generation — Solar Plants (Sampled)</div>', unsafe_allow_html=True)
    solar_gen = fg[fg["plant_type"] == "Solar"].sample(min(3000, len(fg))).copy()
    if not solar_gen.empty:
        solar_gen = solar_gen.merge(fp[["plant_id", "plant_name"]], on="plant_id", how="left")
        fig_scatter = px.scatter(
            solar_gen, x="irradiance_wm2", y="actual_generation_mw",
            color="plant_name", opacity=0.4,
            labels={"irradiance_wm2": "Irradiance (W/m²)", "actual_generation_mw": "Generation (MW)"},
            template="plotly_dark",
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(t=5, b=5), legend_title="Plant",
            legend=dict(font=dict(size=8)),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No solar data for this filter combination.")

    # ── Raw table ────────────────────────────────────────────────────────────
    with st.expander("📋 Plant Master Table"):
        st.dataframe(fp, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — WEATHER REPORT
# ════════════════════════════════════════════════════════════════════════════
elif page == "🌦️ Weather Report":

    import requests
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from datetime import date, timedelta, datetime
    import json

    st.title("🌦️ Weather Report")
    st.caption("Historical · Current · 7-day Forecast — powered by Open-Meteo (free, no API key)")

    # ── Plant locations for quick selection ───────────────────────────────────
    LOCATIONS = {
        "Koppal (Solar)":    (15.35, 76.15),
        "Raichur (Wind)":    (16.20, 77.35),
        "Gadag (Hybrid)":   (15.43, 75.63),
        "Bidar (Solar)":     (17.91, 76.82),
        "Ballari (Wind)":    (14.46, 76.92),
        "Bengaluru":         (12.97, 77.59),
        "Custom location":   None,
    }

    with st.sidebar:
        st.markdown("### 📍 Location")
        loc_choice = st.selectbox("Select location", list(LOCATIONS.keys()))
        if LOCATIONS[loc_choice] is None:
            lat = st.number_input("Latitude", value=15.3, format="%.4f")
            lon = st.number_input("Longitude", value=76.8, format="%.4f")
        else:
            lat, lon = LOCATIONS[loc_choice]
            st.caption(f"Lat: {lat}  |  Lon: {lon}")

        st.markdown("### 📅 Historical Range")
        hist_start = st.date_input("From", value=date.today() - timedelta(days=30))
        hist_end   = st.date_input("To",   value=date.today() - timedelta(days=1))

    # ── Open-Meteo API calls ──────────────────────────────────────────────────
    WMO_CODES = {
        0: ("Clear sky", "☀️"), 1: ("Mainly clear", "🌤️"), 2: ("Partly cloudy", "⛅"),
        3: ("Overcast", "☁️"), 45: ("Foggy", "🌫️"), 48: ("Rime fog", "🌫️"),
        51: ("Light drizzle", "🌦️"), 53: ("Moderate drizzle", "🌦️"), 55: ("Dense drizzle", "🌧️"),
        61: ("Slight rain", "🌧️"), 63: ("Moderate rain", "🌧️"), 65: ("Heavy rain", "🌧️"),
        71: ("Slight snow", "🌨️"), 73: ("Moderate snow", "🌨️"), 75: ("Heavy snow", "❄️"),
        80: ("Slight showers", "🌦️"), 81: ("Moderate showers", "🌧️"), 82: ("Violent showers", "⛈️"),
        85: ("Snow showers", "🌨️"), 95: ("Thunderstorm", "⛈️"), 99: ("Thunderstorm+hail", "⛈️"),
    }

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_historical(lat, lon, start, end):
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": str(start), "end_date": str(end),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                     "windspeed_10m_max,shortwave_radiation_sum",
            "timezone": "Asia/Kolkata",
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        d = r.json()["daily"]
        df = pd.DataFrame(d)
        df["time"] = pd.to_datetime(df["time"])
        return df

    @st.cache_data(ttl=1800, show_spinner=False)
    def fetch_current_and_forecast(lat, lon):
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,"
                       "weather_code,apparent_temperature",
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,"
                      "precipitation,shortwave_radiation",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                     "windspeed_10m_max,shortwave_radiation_sum,weather_code",
            "timezone": "Asia/Kolkata",
            "forecast_days": 8,
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    # Fetch data
    with st.spinner("Fetching weather data from Open-Meteo…"):
        try:
            hist_df = fetch_historical(lat, lon, hist_start, hist_end)
            fc_data = fetch_current_and_forecast(lat, lon)
            fetch_ok = True
        except Exception as e:
            st.error(f"Failed to fetch weather data: {e}")
            fetch_ok = False

    if fetch_ok:
        current = fc_data["current"]
        fc_daily = pd.DataFrame(fc_data["daily"])
        fc_daily["time"] = pd.to_datetime(fc_daily["time"])

        fc_hourly = pd.DataFrame(fc_data["hourly"])
        fc_hourly["time"] = pd.to_datetime(fc_hourly["time"])

        wmo = current.get("weather_code", 0)
        wmo_desc, wmo_icon = WMO_CODES.get(wmo, ("Unknown", "🌡️"))

        # ── Current conditions ─────────────────────────────────────────────
        st.markdown('<div class="section-header">Current Conditions</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(f"{wmo_icon} Conditions", wmo_desc)
        c2.metric("🌡️ Temperature", f"{current['temperature_2m']}°C",
                  f"Feels {current['apparent_temperature']}°C")
        c3.metric("💧 Humidity", f"{current['relative_humidity_2m']}%")
        c4.metric("💨 Wind", f"{current['wind_speed_10m']} km/h")
        c5.metric("📍 Location", loc_choice)

        st.divider()

        # ── 7-day forecast cards ───────────────────────────────────────────
        st.markdown('<div class="section-header">7-Day Forecast</div>', unsafe_allow_html=True)
        fc7 = fc_daily.head(7)
        cols = st.columns(7)
        for i, (_, row) in enumerate(fc7.iterrows()):
            wc = int(row.get("weather_code", 0))
            _, icon = WMO_CODES.get(wc, ("", "🌡️"))
            with cols[i]:
                day_label = row["time"].strftime("%a\n%d %b")
                st.markdown(f"""
                <div class="weather-card">
                  <div style="font-size:0.75rem;color:#aaa;">{row['time'].strftime('%a, %d %b')}</div>
                  <div style="font-size:2rem;">{icon}</div>
                  <div style="font-size:1rem;font-weight:700;">{row['temperature_2m_max']:.0f}°</div>
                  <div style="font-size:0.8rem;color:#888;">{row['temperature_2m_min']:.0f}°</div>
                  <div style="font-size:0.75rem;color:#4fc3f7;">💧 {row['precipitation_sum']:.1f}mm</div>
                  <div style="font-size:0.75rem;color:#a8d8a8;">💨 {row['windspeed_10m_max']:.0f} km/h</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # ── Historical + Forecast temperature chart ────────────────────────
        st.markdown('<div class="section-header">Temperature: Historical → Forecast</div>', unsafe_allow_html=True)

        fig_temp = go.Figure()
        # Historical max/min band
        fig_temp.add_trace(go.Scatter(
            x=hist_df["time"], y=hist_df["temperature_2m_max"],
            name="Hist Max", line=dict(color="#f5a623", width=1.5),
        ))
        fig_temp.add_trace(go.Scatter(
            x=hist_df["time"], y=hist_df["temperature_2m_min"],
            name="Hist Min", line=dict(color="#f5a623", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(245,166,35,0.12)",
        ))
        # Forecast max/min
        fig_temp.add_trace(go.Scatter(
            x=fc_daily["time"], y=fc_daily["temperature_2m_max"],
            name="Fcst Max", line=dict(color="#4fc3f7", width=2),
        ))
        fig_temp.add_trace(go.Scatter(
            x=fc_daily["time"], y=fc_daily["temperature_2m_min"],
            name="Fcst Min", line=dict(color="#4fc3f7", width=1.5, dash="dot"),
            fill="tonexty", fillcolor="rgba(79,195,247,0.12)",
        ))
        # Today marker
        # fig_temp.add_vline(
        #     x=datetime.combine(date.today(), datetime.min.time()), line_dash="dash",
        #     line_color="#e74c3c", annotation_text="Today",
        # )
        fig_temp.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=280,
            margin=dict(t=10, b=10), legend=dict(orientation="h"),
            xaxis_title="", yaxis_title="Temperature (°C)",
        )
        st.plotly_chart(fig_temp, use_container_width=True)

        # ── Precipitation + Solar radiation ────────────────────────────────
        col_p, col_s = st.columns(2)

        with col_p:
            st.markdown('<div class="section-header">Precipitation (Historical + Forecast)</div>', unsafe_allow_html=True)
            all_precip = pd.concat([
                hist_df[["time", "precipitation_sum"]].assign(source="Historical"),
                fc_daily[["time", "precipitation_sum"]].assign(source="Forecast"),
            ])
            fig_rain = px.bar(
                all_precip, x="time", y="precipitation_sum", color="source",
                labels={"precipitation_sum": "Rain (mm)", "time": ""},
                color_discrete_map={"Historical": "#7eb8f7", "Forecast": "#a8d8a8"},
                template="plotly_dark",
            )
            fig_rain.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=240, margin=dict(t=5, b=5), legend_title="",
            )
            st.plotly_chart(fig_rain, use_container_width=True)

        with col_s:
            st.markdown('<div class="section-header">Solar Radiation (Historical + Forecast)</div>', unsafe_allow_html=True)
            all_rad = pd.concat([
                hist_df[["time", "shortwave_radiation_sum"]].assign(source="Historical"),
                fc_daily[["time", "shortwave_radiation_sum"]].assign(source="Forecast"),
            ])
            fig_rad = px.line(
                all_rad, x="time", y="shortwave_radiation_sum", color="source",
                labels={"shortwave_radiation_sum": "Radiation (MJ/m²)", "time": ""},
                color_discrete_map={"Historical": "#f5a623", "Forecast": "#ff6b6b"},
                template="plotly_dark",
            )
            fig_rad.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=240, margin=dict(t=5, b=5), legend_title="",
            )
            st.plotly_chart(fig_rad, use_container_width=True)

        # ── Hourly forecast (next 48h) ─────────────────────────────────────
        st.markdown('<div class="section-header">Hourly Forecast — Next 48 Hours</div>', unsafe_allow_html=True)
        now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
        h48 = fc_hourly[
            (fc_hourly["time"] >= now) &
            (fc_hourly["time"] <= now + pd.Timedelta(hours=48))
        ]

        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            x=h48["time"], y=h48["temperature_2m"],
            name="Temp (°C)", line=dict(color="#f5a623", width=2),
            yaxis="y1",
        ))
        fig_h.add_trace(go.Bar(
            x=h48["time"], y=h48["precipitation"],
            name="Rain (mm)", marker_color="#4fc3f7", opacity=0.6,
            yaxis="y2",
        ))
        fig_h.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=260,
            margin=dict(t=5, b=5), legend=dict(orientation="h"),
            yaxis=dict(title="Temp (°C)", side="left"),
            yaxis2=dict(title="Rain (mm)", overlaying="y", side="right"),
        )
        st.plotly_chart(fig_h, use_container_width=True)

        # ── Wind speed timeline ────────────────────────────────────────────
        st.markdown('<div class="section-header">Wind Speed (Historical + Forecast Max)</div>', unsafe_allow_html=True)
        all_wind = pd.concat([
            hist_df[["time", "windspeed_10m_max"]].assign(source="Historical"),
            fc_daily[["time", "windspeed_10m_max"]].assign(source="Forecast"),
        ])
        fig_wind = px.line(
            all_wind, x="time", y="windspeed_10m_max", color="source",
            labels={"windspeed_10m_max": "Max Wind (km/h)", "time": ""},
            color_discrete_map={"Historical": "#a8d8a8", "Forecast": "#ff6b6b"},
            template="plotly_dark",
        )
        fig_wind.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=220, margin=dict(t=5, b=5), legend_title="",
        )
        st.plotly_chart(fig_wind, use_container_width=True)

        # ── Data table ────────────────────────────────────────────────────
        with st.expander("📋 Raw forecast data"):
            st.dataframe(fc_daily.rename(columns={
                "time": "Date", "temperature_2m_max": "Max Temp (°C)",
                "temperature_2m_min": "Min Temp (°C)",
                "precipitation_sum": "Rain (mm)",
                "windspeed_10m_max": "Max Wind (km/h)",
                "shortwave_radiation_sum": "Solar Rad (MJ/m²)",
                "weather_code": "WMO Code",
            }), use_container_width=True, hide_index=True)

        st.caption("Data: Open-Meteo.com — Free, no API key · Updated hourly · Timezone: Asia/Kolkata")