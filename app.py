import streamlit as st
import pandas as pd

# Page settings
st.set_page_config(
    page_title="Healthcare Risk Dashboard",
    layout="wide"
)

# Title
st.title("🏥 Healthcare Risk Dashboard")

# Load data
df = pd.read_csv("Diabetes_data.csv")

# Sidebar
page = st.sidebar.selectbox(
    "Select Section",
    [
        "Overview",
        "Dataset Preview"
    ]
)

# Overview Page
if page == "Overview":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Patients", len(df))

    with col2:
        st.metric("Average BMI", round(df["BMI"].mean(), 2))

    with col3:
        st.metric("Average Age Category", round(df["Age"].mean(), 2))

# Dataset Preview
elif page == "Dataset Preview":

    st.header("Dataset Preview")

    st.dataframe(df.head())