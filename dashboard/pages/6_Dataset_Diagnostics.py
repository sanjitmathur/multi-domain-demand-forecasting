"""Dataset Diagnostics — distributions, missing values, correlations."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

st.title("Dataset Diagnostics")
st.caption("Data quality checks, distributions, and correlation analysis per domain")

datasets = {
    "airline": ("airline_bookings.csv", "bookings"),
    "ecommerce": ("ecommerce_demand.csv", "quantity_sold"),
    "payment": ("payment_volume.csv", "volume"),
}

domain = st.selectbox("Domain", list(datasets.keys()))
fname, target_col = datasets[domain]
df = pd.read_csv(DATA_DIR / fname)

# --- Overview ---
st.subheader("Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{len(df.columns)}")
c3.metric("Missing Cells", f"{df.isna().sum().sum():,}")
c4.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

# --- Missing Values ---
missing = df.isna().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    st.subheader("Missing Values")
    st.dataframe(pd.DataFrame({"Column": missing.index, "Missing": missing.values,
                                "% Missing": (missing.values / len(df) * 100).round(2)}),
                 hide_index=True)
else:
    st.success("No missing values detected.")

# --- Numeric Summary ---
st.subheader("Numeric Summary")
desc = df.describe().T.round(3)
st.dataframe(desc, use_container_width=True)

# --- Target Distribution ---
st.subheader(f"Target Distribution: `{target_col}`")
fig = go.Figure()
fig.add_trace(go.Histogram(x=df[target_col], nbinsx=50, marker_color="#4ECDC4", opacity=0.8))
fig.add_vline(x=df[target_col].mean(), line_dash="dash", line_color="#FFD93D",
              annotation_text=f"mean={df[target_col].mean():.1f}")
fig.update_layout(xaxis_title=target_col, yaxis_title="Count", template="plotly_dark", height=350)
st.plotly_chart(fig, use_container_width=True)

# --- Correlation Heatmap (numeric cols only) ---
st.subheader("Feature Correlations with Target")
numeric_df = df.select_dtypes(include=[np.number])
if target_col in numeric_df.columns:
    corr = numeric_df.corr()[target_col].drop(target_col).sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h",
                           marker_color=["#4ECDC4" if v > 0 else "#FF6B6B" for v in corr.values]))
    fig.update_layout(title="Pearson Correlation with Target", template="plotly_dark",
                      height=max(300, len(corr) * 22), margin=dict(l=180))
    st.plotly_chart(fig, use_container_width=True)
