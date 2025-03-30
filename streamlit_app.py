import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv("generated_clinical_notes.csv")

df = load_data()

st.title("🩺 Synthetic Clinical Note Viewer")
st.sidebar.header("🔍 Filter Options")

patient_ids = df["Patient ID"].unique()
selected_id = st.sidebar.selectbox("Select Patient ID", sorted(patient_ids))

search_term = st.sidebar.text_input("Search within clinical notes")
filtered = df[df["Patient ID"] == selected_id]
if search_term:
    filtered = filtered[filtered["Clinical Note"].str.contains(search_term, case=False, na=False)]

st.subheader(f"📄 Notes for Patient ID {selected_id}")
for i, row in filtered.iterrows():
    st.markdown(f"**Visit {row['Visit Number']} | Age: {row['Age']} | Gender: {row['Gender']}**")
    st.text_area("Clinical Note", value=row["Clinical Note"], height=300, key=f"note_{i}")
    st.markdown("---")

st.caption("Demo app for viewing synthetic clinical notes (Small Cell Lung Cancer).")
