import streamlit as st
import pandas as pd
import requests
import pydeck as pdk
import numpy as np
import plotly.express as px
import datetime
import json
from model import load_model
from predictor import predict_duration, DEVICE

# Load external JSON files
with open("problem_types.json", encoding="utf-8") as f:
    problem_types = json.load(f)["problem_types"]

with open("areas.json", encoding="utf-8") as f:
    areas = json.load(f)["areas"]

with open("organizations.json", encoding="utf-8") as f:
    organizations = json.load(f)["organizations"]

# ---------- Page Config ----------
st.set_page_config(page_title="Traffy Fondue Dashboard", layout="wide")
st.title("📊 Traffy Fondue: Duration Analysis & Prediction")
st.markdown("""
Welcome to the **Traffy Fondue Dashboard**, a data science project analyzing and predicting issue resolution times from municipal service reports.
""")

# ---------- Sidebar Filters ----------
st.sidebar.header("🔎 Filter Options")
top_n_types = st.sidebar.slider("Top N Problem Types", min_value=1, max_value=50, value=10)
top_n_orgs = st.sidebar.slider("Top N Organizations", min_value=1, max_value=200, value=10)

# ---------- Bar Chart: Average Duration by Problem Type ----------
st.header("📌 Average Fix Duration by Problem Type")
st.caption("This chart shows average issue resolution time per problem type (in days).")

bar_url = "https://njqqsrafv7.execute-api.us-east-1.amazonaws.com/development/duration/avg-by-problem-type"
bar_data = pd.DataFrame(requests.get(bar_url).json()["data"])
st.dataframe(bar_data, use_container_width=True)

if {"problem_type", "duration_minutes"}.issubset(bar_data.columns):
    bar_data = bar_data.dropna(subset=["problem_type"])
    bar_data["duration_days"] = bar_data["duration_minutes"] / 1440
    bar_top = bar_data.sort_values("duration_days", ascending=False).head(top_n_types)

    fig = px.bar(
        bar_top,
        x="problem_type",
        y="duration_days",
        text=bar_top["duration_days"].round(2),
        labels={"duration_days": "Average Duration (days)", "problem_type": "Problem Type"},
        title="Top Problem Types by Average Fix Duration",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Missing required columns in bar chart data.")

# ---------- Box Plot: Duration by Organization ----------
st.header("🏢 Duration Distribution by Organization")
st.caption("Distribution of fix durations by organization. Outliers may indicate systemic delays.")

box_url = "https://njqqsrafv7.execute-api.us-east-1.amazonaws.com/development/duration/boxplot-by-org"
box_data = pd.DataFrame(requests.get(box_url).json()["data"])
st.dataframe(box_data, use_container_width=True)

if {"org", "min_duration", "q1", "median", "q3", "max_duration"}.issubset(box_data.columns):
    box_data = box_data.dropna(subset=["org"])
    box_data = box_data[box_data["org"].str.strip() != ""]
    box_data = box_data.sort_values("median", ascending=False).head(top_n_orgs)

    mock_rows = []
    for _, row in box_data.iterrows():
        org = row["org"]
        durations = []
        for col in ["min_duration", "q1", "median", "q3", "max_duration"]:
            durations.extend([row[col]] * 5)
        for val in durations:
            mock_rows.append({"org": org, "duration_days": val / 1440})

    fig = px.box(pd.DataFrame(mock_rows), x="org", y="duration_days", points="all",
                 labels={"duration_days": "Duration (days)", "org": "Organization"},
                 title="Fix Duration Distribution Across Organizations")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Missing required columns in box plot data.")

# ---------- Heatmap ----------
st.header("🌍 Spatial Heatmap: Average Duration by Location")
st.caption("This heatmap visualizes geographic hotspots where issue resolution takes longer.")

heatmap_url = "https://njqqsrafv7.execute-api.us-east-1.amazonaws.com/development/duration/avg-by-district"
heatmap_data = pd.DataFrame(requests.get(heatmap_url).json()["data"])
st.dataframe(heatmap_data, use_container_width=True)

if {"district", "problem_type", "avg_duration_minutes"}.issubset(heatmap_data.columns):
    if "lat" not in heatmap_data or "lon" not in heatmap_data:
        np.random.seed(0)
        heatmap_data["lat"] = 13.7 + np.random.rand(len(heatmap_data)) * 0.2
        heatmap_data["lon"] = 100.4 + np.random.rand(len(heatmap_data)) * 0.2

    heatmap_data["avg_duration_days"] = heatmap_data["avg_duration_minutes"] / 1440
    view_state = pdk.ViewState(latitude=heatmap_data["lat"].mean(), longitude=heatmap_data["lon"].mean(), zoom=10, pitch=40)
    layer = pdk.Layer("HeatmapLayer", data=heatmap_data, get_position='[lon, lat]', get_weight='avg_duration_days', radiusPixels=60)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
else:
    st.warning("Missing columns in heatmap dataset.")

# ---------- Prediction Form ----------
st.header("🔮 Predict Resolution Time")
st.markdown("Fill in the form below to predict how long an issue might take to resolve.")

@st.cache_resource
def get_model():
    return load_model(
        model_path="best_model_state.bin",
        model_name="airesearch/wangchanberta-base-att-spm-uncased",
        num_org_features=11,
        device=DEVICE,
    )

model = get_model()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        problem_type = st.selectbox("📌 Problem Type", problem_types)
        comment = st.text_area("📝 Problem Details", "มีขยะสะสมที่หน้าบ้านจำนวนมาก ส่งกลิ่นเหม็น")
        organization = st.selectbox("🏢 Organization", organizations)

    with col2:
        timestamp = st.date_input("📅 Report Date", value=datetime.date.today())
        area = st.selectbox("📍 Area", areas)

    submitted = st.form_submit_button("🔍 Predict")
    if submitted:
        try:
            pred_days = predict_duration(
                model=model,
                text=problem_type + " " + comment,
                org=organization,
                comment=comment,
                type_=problem_type,
                timestamp=timestamp,
            )
            st.success(f"🟢 Estimated resolution time: **{pred_days:.2f} days**")
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
