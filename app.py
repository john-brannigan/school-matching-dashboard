import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="EducatEd Choice Dashboard", layout="wide")

st.markdown("""
<style>
.main-title {
    font-size: 44px;
    font-weight: 800;
    color: #1f2a44;
    margin-bottom: 5px;
}
.subtitle {
    font-size: 18px;
    color: #5f6b7a;
    margin-bottom: 30px;
}
.card {
    background-color: #f8fafc;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    margin-bottom: 20px;
}
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.04);
}
.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #1f2a44;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">EducatEd Choice Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Interactive school matching dashboard based on academic performance and stability.</div>',
    unsafe_allow_html=True
)

df = pd.read_csv("master_school_table_v5_2023_24.csv")

features = [
    "grad_rate", "cohort_size", "sat_total", "mobility_rate",
    "mobility_count", "discipline_percent", "hope_eligible_percent"
]

df_pca = df[features].copy()
df_pca = df_pca.fillna(df_pca.median())

X_scaled = StandardScaler().fit_transform(df_pca)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
pca_df["school_name"] = df["school_name"]
pca_df["grad_rate"] = df["grad_rate"]
pca_df["sat_total"] = df["sat_total"]
pca_df["hope_eligible_percent"] = df["hope_eligible_percent"]
pca_df["mobility_rate"] = df["mobility_rate"]

def assign_profile(row):
    if row["PC1"] >= 0 and row["PC2"] >= 0:
        return "High Performance + Stable"
    elif row["PC1"] >= 0 and row["PC2"] < 0:
        return "High Performance + Less Stable"
    elif row["PC1"] < 0 and row["PC2"] >= 0:
        return "Lower Performance + Stable"
    return "Lower Performance + Less Stable"

pca_df["school_profile"] = pca_df.apply(assign_profile, axis=1)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Schools", f"{len(df):,}")
col2.metric("Avg Graduation Rate", f"{df['grad_rate'].mean():.1f}%")
col3.metric("Avg SAT", f"{df['sat_total'].mean():.0f}")
col4.metric("Avg Mobility Rate", f"{df['mobility_rate'].mean():.1f}%")

st.markdown('<div class="section-title">School Matching Map</div>', unsafe_allow_html=True)

fig = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="grad_rate",
    hover_name="school_name",
    hover_data={
        "school_profile": True,
        "grad_rate": ":.1f",
        "sat_total": ":,.0f",
        "hope_eligible_percent": ":.1f",
        "mobility_rate": ":.1f",
        "PC1": False,
        "PC2": False
    },
    labels={
        "PC1": "Academic Performance",
        "PC2": "Stability / Structure",
        "grad_rate": "Graduation Rate",
        "school_profile": "Profile",
        "sat_total": "SAT Total",
        "hope_eligible_percent": "HOPE Eligible (%)",
        "mobility_rate": "Mobility Rate (%)"
    },
    color_continuous_scale="Viridis",
    opacity=0.82
)

fig.add_hline(y=0, line_dash="dash", line_color="gray")
fig.add_vline(x=0, line_dash="dash", line_color="gray")

fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))

fig.update_layout(
    height=620,
    margin=dict(l=20, r=20, t=40, b=20),
    template="plotly_white",
    coloraxis_colorbar=dict(title="Graduation<br>Rate")
)

st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([1.2, 1])

with left:
    st.markdown('<div class="section-title">How to Read This Map</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <b>Academic Performance</b> is shown on the horizontal axis. Schools farther right tend to perform stronger academically.<br><br>
    <b>Stability / Structure</b> is shown on the vertical axis. Schools higher on the map tend to show stronger stability patterns.<br><br>
    The top-right area represents schools that combine stronger performance with stronger stability.
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    • PC1 captures the main academic performance pattern.<br>
    • PC2 highlights stability and structural differences.<br>
    • This map helps connect parent priorities to school profiles.
    </div>
    """, unsafe_allow_html=True)
