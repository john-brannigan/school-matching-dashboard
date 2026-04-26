import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="EducatEd Choice Dashboard", layout="wide")

st.markdown("""
<style>
.main-title {font-size: 42px; font-weight: 800; color: #1f2a44;}
.subtitle {font-size: 17px; color: #5f6b7a; margin-bottom: 25px;}
.card {
    background-color: #f8fafc;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    margin-bottom: 18px;
    line-height: 1.6;
}
.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">EducatEd Choice: School Matching Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Helping families find the best-fit school using data-driven insights.</div>',
    unsafe_allow_html=True
)

df = pd.read_csv("master_school_table_v5_2023_24.csv")

st.sidebar.title("Navigation")
view = st.sidebar.radio(
    "Select Section",
    ["School Matching Map", "PCA Analysis", "Dataset"]
)

features = [
    "grad_rate", "cohort_size", "sat_total", "mobility_rate",
    "mobility_count", "discipline_percent", "hope_eligible_percent"
]

df_pca = df[features].copy()
df_pca = df_pca.fillna(df_pca.median())

X_scaled = StandardScaler().fit_transform(df_pca)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
cum_var = explained_var.cumsum()

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
col3.metric("Avg SAT Score", f"{df['sat_total'].mean():.0f}")
col4.metric("Avg Mobility Rate", f"{df['mobility_rate'].mean():.1f}%")

if view == "School Matching Map":
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
        opacity=0.82,
        title="School Matching Map: Performance vs Stability"
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
    fig.update_layout(height=620, template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="card">
    <b>How this map helps match schools:</b><br><br>
    This chart combines many school factors into two simple ideas:
    <b>academic performance</b> and <b>school stability/structure</b>.<br><br>
    Instead of comparing schools one column at a time, this map shows the big picture:
    <br>• Right side → stronger academic outcomes
    <br>• Top side → stronger stability pattern
    <br>• Top-right area → stronger overall match<br><br>
    This supports the main goal of EducatEd Choice: turning complex school data into a simple tool families can use to compare schools based on what matters most to them.
    </div>
    """, unsafe_allow_html=True)

elif view == "PCA Analysis":
    st.markdown('<div class="section-title">PCA Scatter Plot</div>', unsafe_allow_html=True)

    fig_scatter = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="grad_rate",
        hover_name="school_name",
        labels={
            "PC1": f"PC1 ({explained_var[0]*100:.1f}% variance)",
            "PC2": f"PC2 ({explained_var[1]*100:.1f}% variance)",
            "grad_rate": "Graduation Rate"
        },
        color_continuous_scale="Viridis",
        title="PCA Scatter Plot of Schools (PC1 vs PC2)"
    )

    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_scatter.update_traces(marker=dict(size=8, line=dict(width=0.5, color="black")))
    fig_scatter.update_layout(height=600, template="plotly_white")

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("""
    <div class="card">
    <b>What this chart shows:</b><br><br>
    Each point represents one school. Schools with similar academic and structural patterns are placed closer together,
    while schools with different patterns are farther apart.<br><br>
    The color represents graduation rate. The right side of the chart generally contains schools with stronger graduation outcomes,
    which shows that <b>PC1 is closely connected to academic performance</b>.<br><br>
    This is important because the matching system needs a fair way to compare schools using several variables at once, not just one score.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key Drivers of School Differences</div>', unsafe_allow_html=True)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i+1}" for i in range(len(features))]
    )

    pc1_loadings = loadings["PC1"].sort_values(key=abs)
    loadings_df = pc1_loadings.reset_index()
    loadings_df.columns = ["Feature", "Contribution"]

    def feature_group(feature):
        if feature in ["grad_rate", "sat_total", "hope_eligible_percent"]:
            return "Academic Performance"
        elif feature == "cohort_size":
            return "School Scale"
        else:
            return "School Structure"

    loadings_df["Group"] = loadings_df["Feature"].apply(feature_group)

    fig_loadings = px.bar(
        loadings_df,
        x="Contribution",
        y="Feature",
        color="Group",
        orientation="h",
        title="Key Drivers of School Differences",
        color_discrete_map={
            "Academic Performance": "blue",
            "School Structure": "orange",
            "School Scale": "gray"
        }
    )

    fig_loadings.update_layout(
        height=520,
        template="plotly_white",
        xaxis_title="Contribution Strength",
        yaxis_title="Features"
    )

    st.plotly_chart(fig_loadings, use_container_width=True)

    st.markdown("""
    <div class="card">
    <b>Why this chart matters:</b><br><br>
    This chart shows which variables are most responsible for separating schools in the PCA model.<br><br>
    Academic factors like graduation rate, SAT score, and HOPE eligibility are important, but the chart also shows that school size and mobility-related variables play a role.<br><br>
    This supports the client’s matching idea because a good school match should not only consider test scores.
    It should also consider the school environment, size, and stability.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Explained Variance</div>', unsafe_allow_html=True)

    variance_df = pd.DataFrame({
        "Principal Component": [f"PC{i+1}" for i in range(len(explained_var))],
        "Individual Variance": explained_var,
        "Cumulative Variance": cum_var
    })

    fig_var = px.bar(
        variance_df,
        x="Principal Component",
        y="Individual Variance",
        title="Explained Variance by Principal Components",
        text_auto=".1%"
    )

    fig_var.add_scatter(
        x=variance_df["Principal Component"],
        y=variance_df["Cumulative Variance"],
        mode="lines+markers",
        name="Cumulative Variance"
    )

    fig_var.update_layout(
        height=520,
        template="plotly_white",
        yaxis_tickformat=".0%",
        yaxis_title="Variance Explained"
    )

    st.plotly_chart(fig_var, use_container_width=True)

    st.markdown("""
    <div class="card">
    <b>Why PCA is useful here:</b><br><br>
    Schools have many different variables, which can be hard to compare directly.
    PCA simplifies those variables into a smaller number of meaningful patterns.<br><br>
    PC1 explains the largest share of the difference between schools, so it becomes the strongest summary dimension for comparison.
    The cumulative line shows how much information is captured as more components are added.<br><br>
    This helps justify using PCA in the matching system because it keeps the most important information while making the results easier to explain.
    </div>
    """, unsafe_allow_html=True)

elif view == "Dataset":
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    This dataset is the foundation of the dashboard. It includes school-level academic and structural variables such as graduation rate,
    SAT score, mobility, discipline, cohort size, and HOPE eligibility.<br><br>
    These variables are used to create the PCA model and support the school matching logic shown in the dashboard.
    </div>
    """, unsafe_allow_html=True)

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])
    st.dataframe(df.head(50), use_container_width=True)
