"""Streamlit dashboard for multi-domain demand forecasting."""

import streamlit as st

st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide",
                   initial_sidebar_state="expanded")

pg = st.navigation([
    st.Page("pages/0_Forecasting.py", title="Demand Forecasting", icon="📊", default=True),
    st.Page("pages/1_Model_Architecture.py", title="Model Architecture", icon="🏗️"),
    st.Page("pages/2_Model_Evaluation.py", title="Model Evaluation", icon="📈"),
    st.Page("pages/3_Feature_Importance.py", title="Feature Importance", icon="🔍"),
    st.Page("pages/4_Uncertainty_Analysis.py", title="Uncertainty Analysis", icon="📉"),
    st.Page("pages/5_System_Design.py", title="System Design", icon="⚙️"),
    st.Page("pages/6_Dataset_Diagnostics.py", title="Dataset Diagnostics", icon="🧪"),
    st.Page("pages/7_Ablation_Study.py", title="Ablation Study", icon="🔬"),
])

pg.run()
