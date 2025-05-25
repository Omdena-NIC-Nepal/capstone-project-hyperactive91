import streamlit as st
import pandas as pd
import plotly.express as px
from utils.pred import (
    get_climate_forecast,
    get_highheat_forecast,
    get_drought_forecast,
    get_glacier_forecast
)

def predict():
    st.title("Climate Forecast")
    st.markdown("Explore and interact with forecasted climate indicators for Nepal from 2020 to 2050 using dynamic visualizations and data-driven insights.")

    # --- Forecast Category Selection ---
    option = st.selectbox(
        "Select forecast :",
        ["Avg. Temperature", "Highheat Days", "Risk of Drought", "Glacier melting"]
    )

    # --- Forecast Viewer Logic ---
    if option == "Avg. Temperature":
        df = get_climate_forecast()
        district = st.selectbox("Select District", sorted(df["District"].unique()))
        df_d = df[df["District"] == district]

        fig = px.line(
            df_d,
            x="YEAR", y="predicted_avg_temp",
            title=f"Forecasted Avg Temperature in {district} (2020–2050)",
            labels={"predicted_avg_temp": "Temperature (°C)", "YEAR": "Year"}
        )
        fig.update_traces(line=dict(color='orange'))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_d)

    elif option == "Highheat Days":
        df = get_highheat_forecast()
        district = st.selectbox("Select District", sorted(df["District"].unique()))
        df_d = df[df["District"] == district]

        fig = px.area(
            df_d,
            x="YEAR", y="predicted_highheat_days",
            title=f"Forecasted Highheat Days in {district} (2020–2050)",
            labels={"predicted_highheat_days": "Days >38°C", "YEAR": "Year"}
        )
        fig.update_traces(line=dict(color='red'))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_d)

    elif option == "Risk of Drought":
        df = get_drought_forecast()
        district = st.selectbox("Select District", sorted(df["District"].unique()))
        df_d = df[df["District"] == district]

        fig = px.line(
            df_d,
            x="YEAR", y="predicted_spi",
            title=f"SPI-based Drought Forecast in {district} (2020–2050)",
            labels={"predicted_spi": "SPI (z-score)", "YEAR": "Year"},
        )
        fig.update_traces(line=dict(color='brown'))
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Forecasted Drought Categories")
        st.dataframe(df_d[["YEAR", "predicted_spi", "drought_risk"]])

    elif option == "Glacier melting":
        df = get_glacier_forecast()
        subbasin = st.selectbox("Select Sub-Basin", sorted(df["sub-basin"].unique()))
        df_s = df[df["sub-basin"] == subbasin]

        fig = px.area(
            df_s,
            x="year", y="predicted_glacier_area",
            title=f"Forecasted Glacier Area in {subbasin} (2020–2050)",
            labels={"predicted_glacier_area": "Glacier Area (km²)", "year": "Year"}
        )
        fig.update_traces(line=dict(color='purple'))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Ice Volume & Elevation Forecast")
        st.dataframe(df_s[["year", "predicted_ice_volume", "predicted_min_elev"]].round(2))
