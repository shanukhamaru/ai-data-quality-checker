import streamlit as st
import pandas as pd
from data_checker import basic_stats, missing_values, duplicate_rows, constant_columns
from langchain_agent import get_cleaning_suggestions
from dotenv import load_dotenv
import json

# Load API keys and env variables
load_dotenv()

# App config and title
st.set_page_config(page_title="AI Data Quality Checker", layout="wide")
st.title("🧪 AI Data Quality Checker")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Info")
    st.write(f"Shape: {df.shape}")
    st.write("Columns:", list(df.columns))

    st.subheader("📋 Data Quality Report")

    with st.expander("🔹 Basic Stats"):
        st.json(basic_stats(df))

    with st.expander("🔸 Missing Values"):
        st.json(missing_values(df))

    with st.expander("🔹 Duplicate Rows"):
        st.json(duplicate_rows(df))

    with st.expander("🔸 Constant/Near-Constant Columns"):
        st.json(constant_columns(df))

    # 🔍 Collect issues to send to AI
    issues = {}

    mv = missing_values(df)
    dup = duplicate_rows(df)
    const_cols = constant_columns(df)

    if any(v > 0 for v in mv.values()):
        issues["Missing Values"] = {k: v for k, v in mv.items() if v > 0}

    if dup["duplicate_row_count"] > 0:
        issues["Duplicate Rows"] = dup["duplicate_row_count"]

    if const_cols:
        issues["Constant Columns"] = const_cols

    # 🤖 AI Suggestions Section
    st.subheader("🤖 AI Cleaning Suggestions")
    if issues:
        with st.spinner("Analyzing with LLM..."):
            suggestions = get_cleaning_suggestions(issues)
            st.write(suggestions)
    else:
        st.success("✅ No major data issues detected.")
    st.subheader("🛠️ Auto Cleaning Options")

    if st.button("🧹 Clean My Data"):
        df_cleaned = df.copy()

        # Drop duplicate rows
        df_cleaned = df_cleaned.drop_duplicates()

        # Drop constant columns
        for col in const_cols:
            df_cleaned.drop(columns=col, inplace=True)

        # Fill missing values
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() > 0:
                if df_cleaned[col].dtype in ['float64', 'int64']:
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                else:
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

        st.success("✅ Data cleaned successfully!")
        st.subheader("📄 Cleaned Data Preview")
        st.dataframe(df_cleaned.head())

        # Download button
        csv = df_cleaned.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv,
            file_name='cleaned_data.csv',
            mime='text/csv'
        )


