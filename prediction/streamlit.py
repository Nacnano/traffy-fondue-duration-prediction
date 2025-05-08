import streamlit as st
import pandas as pd
import requests
import pydeck as pdk
import numpy as np
import plotly.express as px
import datetime

from model import load_model
from predictor import predict_duration, DEVICE

st.set_page_config(page_title="Traffy Fondue Dashboard", layout="wide")

# Sidebar controls
st.sidebar.header("Filter Options")
top_n_types = st.sidebar.slider(
    "Number of Problem Types to Display", min_value=1, max_value=20, value=10
)
top_n_orgs = st.sidebar.slider(
    "Number of Organizations to Display", min_value=1, max_value=20, value=10
)

st.title("Traffy Fondue Dashboard")

# --------- Bar Chart ---------
st.header("Average Fix Duration by Problem Type")
st.markdown(
    """
**What does this chart show?**  
This bar chart displays the **average duration (in days)** it takes to resolve issues for each problem type.  
"""
)
bar_url = "https://njqqsrafv7.execute-api.us-east-1.amazonaws.com/development/duration/avg-by-problem-type"
bar_raw = requests.get(bar_url).json()
bar_data = pd.DataFrame(bar_raw["data"])

st.subheader("Bar Chart API Data")
st.write(bar_data)
# st.write("Columns:", bar_data.columns.tolist())

if "problem_type" in bar_data.columns and "duration_minutes" in bar_data.columns:
    bar_data = bar_data.dropna(subset=["problem_type"])
    bar_data["problem_type"] = bar_data["problem_type"].astype(str)
    bar_data["duration_days"] = (
        bar_data["duration_minutes"] / 1440
    )  # Convert minutes to days

    # st.write("Unique problem types:", bar_data["problem_type"].unique())

    # Sort and take top N
    bar_data = bar_data.sort_values("duration_days", ascending=False).head(top_n_types)

    # Plotly bar chart
    fig = px.bar(
        bar_data,
        x="problem_type",
        y="duration_days",
        text=bar_data["duration_days"].round(2),
        labels={
            "duration_days": "Average Duration (days)",
            "problem_type": "Problem Type",
        },
        title="Average Fix Duration by Problem Type",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(
        "Expected columns 'problem_type' and 'duration_minutes' not found in bar chart data."
    )

# --------- Box Plot (True Box Plot Implementation) ---------
st.header("Duration Distribution by Organization (Box Plot)")
st.markdown(
    """
**What does this chart show?**  
This **box plot** displays the distribution of fix durations (in days) for each organization.  
Outliers might be points of interest to investigate further.
"""
)
box_url = "https://njqqsrafv7.execute-api.us-east-1.amazonaws.com/development/duration/boxplot-by-org"
box_raw = requests.get(box_url).json()
box_data = pd.DataFrame(box_raw["data"])

st.subheader("Box Plot API Data")
st.write(box_data)
# st.write("Columns:", box_data.columns.tolist())

if "org" in box_data.columns and {
    "min_duration",
    "q1",
    "median",
    "q3",
    "max_duration",
}.issubset(box_data.columns):
    box_data = box_data.dropna(subset=["org"])
    box_data = box_data[box_data["org"].str.strip() != ""]  # Remove empty orgs
    box_data = box_data.sort_values("median", ascending=False).head(top_n_orgs)

    # Mock raw data from summary stats for visualization only
    expanded_rows = []
    for _, row in box_data.iterrows():
        org = row["org"]
        synthetic_data = (
            [row["min_duration"]] * 5
            + [row["q1"]] * 5
            + [row["median"]] * 5
            + [row["q3"]] * 5
            + [row["max_duration"]] * 5
        )
        for val in synthetic_data:
            expanded_rows.append({"org": org, "duration_minutes": val})

    mock_data = pd.DataFrame(expanded_rows)
    mock_data["duration_days"] = mock_data["duration_minutes"] / 1440

    # Plotly box plot
    fig = px.box(
        mock_data,
        x="org",
        y="duration_days",
        points="all",
        labels={"duration_days": "Duration (days)", "org": "Organization"},
        title="Duration Distribution by Organization",
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Expected boxplot columns not found in box plot data.")

# --------- PyDeck Heatmap (Spatial) ---------
st.header("Spatial Heatmap: Average Duration by Location")
st.markdown(
    """
**What does this chart show?**  
This spatial heatmap shows where problems take longer to resolve, visualized on a map.  
Each hotspot represents a location, and its intensity reflects the **average duration (in days)**.  
"""
)
heatmap_url = "https://njqqsrafv7.execute-api.us-east-1.amazonaws.com/development/duration/avg-by-district"
heatmap_raw = requests.get(heatmap_url).json()
heatmap_data = pd.DataFrame(heatmap_raw["data"])

st.subheader("Heatmap API Data")
st.write(heatmap_data)
# st.write("Columns:", heatmap_data.columns.tolist())

if {"district", "problem_type", "avg_duration_minutes"}.issubset(heatmap_data.columns):
    # Add mock lat/lon if not present
    if "lat" not in heatmap_data.columns or "lon" not in heatmap_data.columns:
        np.random.seed(0)
        heatmap_data["lat"] = 13.7 + np.random.rand(len(heatmap_data)) * 0.2
        heatmap_data["lon"] = 100.4 + np.random.rand(len(heatmap_data)) * 0.2

    # Convert to days
    heatmap_data["avg_duration_days"] = heatmap_data["avg_duration_minutes"] / 1440

    st.write("Spatial Heatmap (PyDeck)")
    layer = pdk.Layer(
        "HeatmapLayer",
        data=heatmap_data,
        get_position="[lon, lat]",
        get_weight="avg_duration_days",
        radiusPixels=60,
    )

    view_state = pdk.ViewState(
        latitude=heatmap_data["lat"].mean(),
        longitude=heatmap_data["lon"].mean(),
        zoom=10,
        pitch=40,
    )

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

else:
    st.warning(
        "Expected columns 'district', 'problem_type', and 'avg_duration_minutes' not found in heatmap data."
    )

# --------- Prediction Section ---------
st.header("📍 Predict Resolution Time from Your Input")
st.markdown("ลองใส่ข้อมูลด้านล่างเพื่อให้ระบบคาดการณ์ระยะเวลาในการแก้ไขปัญหา")


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
        problem_type = st.selectbox(
            "ประเภทปัญหา",
            [
                "ร้องเรียน",
                "แสงสว่าง",
                "สัตว์จรจัด",
                "สะพาน",
                "ความปลอดภัย",
                "เสนอแนะ",
                "ถนน",
                "ทางเท้า",
                "สายไฟ",
                "คลอง",
                "ท่อระบายน้ำ",
                "จราจร",
                "เสียงรบกวน",
                "การเดินทาง",
                "น้ำท่วม",
                "ต้นไม้",
                "กีดขวาง",
                "คนจรจัด",
                "ความสะอาด",
                "เสียง",
                "สถานบันเทิง",
                "ไฟฟ้า",
                "จุดเสี่ยง",
                "อาคารสถานที่ชำรุด",
                "อุบัติเหตุ",
                "ผิดกฎจราจร",
                "อุปกรณ์ชำรุด",
                "คนเร่ร่อน",
                "ฝุ่นควัน&กลิ่น&PM2.5",
                "เผาในที่โล่ง",
                "หาบเร่แผงลอย",
                "สัตว์",
                "อื่นๆ",
                "ป้ายโฆษณา",
                "ขึ้นทะเบียน&สำรวจ",
                "ประปา",
                "ทุจริต",
                "กรุงเทพโปร่งใส",
                "ขอใช้บริการ",
                "สายสื่อสาร",
                "ขอความช่วยเหลือ",
                "แท็กซี่/รถเมล์",
                "ชื่นชม",
            ],
        )

        comment = st.text_area("รายละเอียดปัญหา", "มีขยะสะสมที่หน้าบ้านจำนวนมาก ส่งกลิ่นเหม็น")
        organization = st.selectbox(
            "หน่วยงานที่เกี่ยวข้อง",
            [
                "Bangkok Smart Lighting (สำนักการโยธา กทม.)",
                "MRTA",
                "Traffy @ ITS Lab2",
                "กลุ่มงานควบคุมอาคาร 2 ส่วนควบคุมอาคาร 2 สำนักงานควบคุมอาคาร (สคอ.สนย.)",
                "กลุ่มงานจัดการระบบราง สขร.สขส.สจส. กทม.",
                "กลุ่มงานระบบขนส่งทางบกและทางน้ำ สบน.สขส.สจส. กทม.",
                "กลุ่มงานวางแผนและออกแบบ 1 สวจ.สจส. กทม.",
                "กลุ่มงานโครงการระบบราง สขส.สจส. กทม.",
                "ทศท",
                "ฟขข.รับร้องเรียนฟุตบาท+ฝาท่อ",
                "รถไฟฟ้าสายสีแดง รฟท.",
                "ศูนย์กำจัดมูลฝอยสายไหม กกฝ. สสล.",
                "ศูนย์อำนวยการบรรเทาสาธารณภัย ส่วนกลาง",
                "สำนักงานสวนสาธารณะ สำนักสิ่งแวดล้อม",
                "ส่วนสวนสาธารณะ 1 สสณ. สสล.",
                "ส่วนสวนสาธารณะ 2 สสณ. สสล.",
                "โครงการถนนวิทยุ",
                "โครงการถนนหลังสวนและถนนสารสิน",
                "โครงการรถไฟฟ้าสายสีเขียวเหนือ (ถนนพหลโยธิน และถนนลาดพร้าว)",
                "โยธาธิการ",
                "Other",
            ],
        )

    with col2:
        timestamp = st.date_input("วันที่แจ้งเรื่อง", value=datetime.date.today())
        area = st.selectbox(
            "พื้นที่",
            [
                "คลองสาน",
                "คลองสามวา",
                "คลองเตย",
                "คันนายาว",
                "จตุจักร",
                "จอมทอง",
                "ดอนเมือง",
                "ดินแดง",
                "ดุสิต",
                "ตลิ่งชัน",
                "ทวีวัฒนา",
                "ทุ่งครุ",
                "ธนบุรี",
                "ธัญบุรี",
                "บางกอกน้อย",
                "บางกอกใหญ่",
                "บางกะปิ",
                "บางขุนเทียน",
                "บางคอแหลม",
                "บางซื่อ",
                "บางนา",
                "บางบอน",
                "บางพลัด",
                "บางพลี",
                "บางรัก",
                "บางเขน",
                "บางแค",
                "บึงกุ่ม",
                "ปทุมวัน",
                "ประเวศ",
                "ป้อมปราบศัตรูพ่าย",
                "พญาไท",
                "พระนคร",
                "พระโขนง",
                "ภาษีเจริญ",
                "มีนบุรี",
                "ยานนาวา",
                "ราชเทวี",
                "ราษฎร์บูรณะ",
                "ลาดกระบัง",
                "ลาดพร้าว",
                "วังทองหลาง",
                "วัฒนา",
                "สวนหลวง",
                "สะพานสูง",
                "สัมพันธวงศ์",
                "สาทร",
                "สายไหม",
                "หนองจอก",
                "หนองแขม",
                "หลักสี่",
                "ห้วยขวาง",
                "เมืองปทุมธานี",
                "เมืองสมุทรปราการ",
            ],
        )

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

            st.success(
                f"ระบบคาดการณ์ว่าอาจใช้เวลา **{pred_days:.2f} วัน** ในการดำเนินการแก้ไขปัญหานี้"
            )
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการพยากรณ์: {e}")

# # --------- List of All Problem Types ---------
# st.subheader("📋 All Problem Types in Dataset")
# if not bar_data.empty and "problem_type" in bar_data.columns:
#     all_problem_types = bar_data["problem_type"].dropna().unique()
#     st.markdown(f"Total problem types: **{len(all_problem_types)}**")
#     for ptype in sorted(all_problem_types):
#         st.markdown(f"- {ptype}")

# # --------- List of All Organizations ---------
# st.subheader("🏢 All Organizations in Dataset")
# if not box_data.empty and "org" in box_data.columns:
#     all_orgs = box_data["org"].dropna().unique()
#     st.markdown(f"Total organizations: **{len(all_orgs)}**")
#     for org in sorted(all_orgs):
#         st.markdown(f"- {org}")

# # --------- List of All Districts ---------
# st.subheader("📍 All Districts in Dataset")
# if not heatmap_data.empty and "district" in heatmap_data.columns:
#     all_districts = heatmap_data["district"].dropna().unique()
#     st.markdown(f"Total districts: **{len(all_districts)}**")
#     for district in sorted(all_districts):
#         st.markdown(f"- {district}")