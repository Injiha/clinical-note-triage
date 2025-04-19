

import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv("generated_clinical_notes.csv")  # Update filename if needed

df = load_data()

# App Title
st.title("🩺 Synthetic Clinical Notes - Table View")

# Display the whole table
st.dataframe(df, use_container_width=True)

st.caption("This table displays all generated clinical notes for analysis and modeling.")

