import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

st.set_page_config(layout="wide")

st.title("🚗 Overspeed Vehicle Detection Dashboard")

# -------------------------------
# Load CSV
# -------------------------------
if not os.path.exists("violations.csv"):
    st.error("violations.csv not found. Run main.py first.")
    st.stop()

df = pd.read_csv("violations.csv")

if df.empty:
    st.warning("No violations recorded yet.")
    st.stop()

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("Filters")

min_speed = st.sidebar.slider("Min Speed", 0, 150, 0)
max_speed = st.sidebar.slider("Max Speed", 0, 200, 200)

filtered_df = df[
    (df["Speed_km_h"] >= min_speed) &
    (df["Speed_km_h"] <= max_speed)
].copy()

# -------------------------------
# KPIs
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Violations", len(filtered_df))
col2.metric("Max Speed", f"{filtered_df['Speed_km_h'].max()} km/h")
col3.metric("Avg Speed", f"{round(filtered_df['Speed_km_h'].mean(),2)} km/h")

st.divider()

# -------------------------------
# Table
# -------------------------------
st.subheader("📊 Violation Data")
st.dataframe(filtered_df, use_container_width=True)

# -------------------------------
# Graphs
# -------------------------------
sns.set_style("whitegrid")

colA, colB = st.columns(2)

with colA:
    st.subheader("📈 Speed vs Time")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x="Timestamp_sec", y="Speed_km_h", data=filtered_df, ax=ax1)
    st.pyplot(fig1)

with colB:
    st.subheader("📉 Speed Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(filtered_df["Speed_km_h"], bins=10, kde=True, ax=ax2)
    st.pyplot(fig2)

# -------------------------------
# Violations over time
# -------------------------------
st.subheader("📊 Violations Over Time")

filtered_df["Time_bin"] = filtered_df["Timestamp_sec"].astype(int)
count_per_time = filtered_df.groupby("Time_bin").size()

fig3, ax3 = plt.subplots()
count_per_time.plot(kind="bar", ax=ax3)
st.pyplot(fig3)

# -------------------------------
# Vehicle Snapshots (SMART MATCH)
# -------------------------------
st.subheader("🚗 Violating Vehicles (Snapshots)")

image_folder = "output/images"
image_files = glob.glob(os.path.join(image_folder, "*.jpg"))

if len(image_files) == 0:
    st.warning("No images found.")
else:
    # Show images in grid
    cols = st.columns(3)

    for i, img_path in enumerate(image_files[:9]):  # limit to 9 images
        with cols[i % 3]:
            st.image(img_path, use_container_width=True)

# -------------------------------
# Video
# -------------------------------
st.subheader("🎥 Output Video")

if os.path.exists("final_output.mp4"):
    st.video("final_output.mp4")
else:
    st.warning("Video not found.")