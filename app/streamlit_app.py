"""
app/streamlit_app.py
====================
Production Streamlit dashboard for the Customer Segmentation system.

Features
--------
  📊 Dashboard   – KPIs, revenue overview, segment distribution
  🔍 Predict     – Score a single customer in real-time
  📋 Batch       – Upload CSV and score many customers
  📈 Insights    – Business recommendations per segment
  📚 Docs        – About the model and methodology
"""

import sys
import warnings
from pathlib import Path
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

# ── sys.path so we can import src ──────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config
from src.predict import CustomerSegmentPredictor, CLUSTER_PROFILES

warnings.filterwarnings("ignore")

# ===========================================================================
# PAGE CONFIG
# ===========================================================================

st.set_page_config(
    page_title="Customer Segmentation · ML Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# LOAD CONFIG & MODEL
# ===========================================================================

@st.cache_resource(show_spinner="Loading ML artefacts …")
def load_predictor():
    cfg = load_config()
    try:
        predictor = CustomerSegmentPredictor(cfg).load()
        return predictor, cfg, True
    except FileNotFoundError:
        return None, cfg, False


@st.cache_data(show_spinner="Loading clustered data …")
def load_clustered_data():
    path = PROJECT_ROOT / "data" / "processed" / "rfm_features_clustered.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0)
        return df
    return None


predictor, cfg, MODEL_LOADED = load_predictor()
cluster_data = load_clustered_data()

# ===========================================================================
# CUSTOM CSS
# ===========================================================================

st.markdown("""
<style>
  /* ── Fonts & base ─────────────────────────────────────────── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── KPI cards ───────────────────────────────────────────── */
  .kpi-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }
  .kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
    margin: 0;
  }
  .kpi-label {
    font-size: 0.82rem;
    color: #8892a4;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
  }

  /* ── Segment badge ───────────────────────────────────────── */
  .segment-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.9rem;
    margin: 8px 0;
  }

  /* ── Strategy list ────────────────────────────────────────── */
  .strategy-item {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #00d4ff;
    padding: 10px 16px;
    margin: 6px 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.9rem;
  }

  /* ── Header strip ─────────────────────────────────────────── */
  .header-strip {
    background: linear-gradient(90deg, #0f3460, #533483, #0f3460);
    padding: 18px 28px;
    border-radius: 12px;
    margin-bottom: 20px;
  }
  .header-strip h1 {
    color: white;
    margin: 0;
    font-size: 1.6rem;
    font-weight: 700;
  }
  .header-strip p {
    color: rgba(255,255,255,0.7);
    margin: 4px 0 0 0;
    font-size: 0.9rem;
  }
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# SIDEBAR NAV
# ===========================================================================

with st.sidebar:
    st.markdown("### 🛍️ Customer Segmentation")
    st.markdown("*Production ML Dashboard*")
    st.divider()

    page = option_menu(
        menu_title=None,
        options=["Dashboard", "Predict Segment", "Batch Scoring",
                 "Segment Insights", "Methodology"],
        icons=["bar-chart-line", "search", "table",
               "lightbulb", "book"],
        default_index=0,
        styles={
            "container": {"padding": "0"},
            "icon": {"font-size": "14px"},
            "nav-link": {"font-size": "14px", "padding": "8px 12px"},
        },
    )

    st.divider()
    if MODEL_LOADED:
        st.success("✅ Model loaded")
    else:
        st.warning("⚠️ Run pipeline first")

    st.markdown(
        """
        <small>
        **Dataset**: UCI Online Retail II<br>
        **Algorithm**: K-Means (k=4)<br>
        **Features**: RFM + 7 advanced<br>
        **Scaler**: RobustScaler
        </small>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# HELPERS
# ===========================================================================

def kpi_card(value: str, label: str) -> str:
    return f"""
    <div class="kpi-card">
      <p class="kpi-value">{value}</p>
      <p class="kpi-label">{label}</p>
    </div>"""


def render_prediction(result: dict) -> None:
    """Render a rich prediction result card."""
    color  = result["color"]
    name   = result["cluster_name"]
    emoji  = result["emoji"]

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{color}22,{color}11);
                border:2px solid {color}55; border-radius:16px; padding:24px;">
      <h2 style="color:{color};margin:0">{emoji} {name}</h2>
      <p style="color:#ccc;margin:8px 0 0 0">{result['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Cluster ID",       result["cluster_id"])
    c2.metric("Retention Risk",   result["retention_risk"])
    c3.metric("Upsell Potential", result["upsell_potential"])

    st.markdown("#### 📌 RFM Profile")
    st.info(result["rfm_profile"])

    st.markdown("#### 🎯 Marketing Strategies")
    for s in result["strategy"]:
        st.markdown(
            f'<div class="strategy-item">• {s}</div>',
            unsafe_allow_html=True,
        )


# ===========================================================================
# PAGE: DASHBOARD
# ===========================================================================

if page == "Dashboard":
    st.markdown("""
    <div class="header-strip">
      <h1>📊 Customer Segmentation Dashboard</h1>
      <p>Real-time cluster analytics powered by K-Means + RFM feature engineering</p>
    </div>
    """, unsafe_allow_html=True)

    if cluster_data is None:
        st.info("👋 Run the training pipeline to populate the dashboard.")
        st.code("python -m src.train_model", language="bash")
        st.stop()

    df = cluster_data.copy()

    # ── KPI Row ─────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.markdown(kpi_card(f"{len(df):,}", "Customers"), unsafe_allow_html=True)
    with k2: st.markdown(kpi_card(f"£{df['Monetary'].sum()/1e6:.2f}M", "Total Revenue"), unsafe_allow_html=True)
    with k3: st.markdown(kpi_card(f"£{df['AvgOrderValue'].mean():.0f}", "Avg Order Value"), unsafe_allow_html=True)
    with k4: st.markdown(kpi_card(f"{df['Recency'].mean():.0f}d", "Avg Recency"), unsafe_allow_html=True)
    with k5: st.markdown(kpi_card(f"{df['Cluster'].nunique()}", "Segments"), unsafe_allow_html=True)

    st.divider()

    # ── Cluster distribution ─────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        seg_counts = df["Cluster_Name"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig = px.pie(
            seg_counts, values="Count", names="Segment",
            title="Customer Distribution by Segment",
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.45,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        rev_by_seg = (
            df.groupby("Cluster_Name")["Monetary"]
            .sum()
            .reset_index()
            .sort_values("Monetary", ascending=True)
        )
        fig = px.bar(
            rev_by_seg, x="Monetary", y="Cluster_Name",
            orientation="h",
            title="Total Revenue by Segment",
            color="Monetary",
            color_continuous_scale="Viridis",
            labels={"Monetary": "Revenue (£)", "Cluster_Name": ""},
        )
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── RFM scatter ──────────────────────────────────────────────────────
    st.markdown("#### RFM Scatter: Frequency vs Monetary (sized by Recency)")
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="Frequency", y="Monetary",
        color="Cluster_Name",
        size=np.clip(1 / (sample["Recency"] + 1) * 300, 2, 20),
        hover_data=["Recency", "AvgOrderValue", "CLV"],
        title="",
        opacity=0.65,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # ── Cluster feature summary table ────────────────────────────────────
    st.markdown("#### Cluster Profiles — Key Metrics")
    summary = (
        df.groupby("Cluster_Name")
        .agg(
            Count=("Recency", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Avg_CLV=("CLV", "mean"),
            Avg_AOV=("AvgOrderValue", "mean"),
        )
        .round(1)
        .sort_values("Avg_Monetary", ascending=False)
        .reset_index()
    )
    st.dataframe(
        summary.style.background_gradient(
            cmap="YlGn", subset=["Avg_Monetary", "Avg_CLV"]
        ).format({
            "Avg_Recency": "{:.0f}d",
            "Avg_Frequency": "{:.1f}",
            "Avg_Monetary": "£{:,.0f}",
            "Avg_CLV": "£{:,.0f}",
            "Avg_AOV": "£{:,.0f}",
        }),
        use_container_width=True,
        height=200,
    )


# ===========================================================================
# PAGE: PREDICT SEGMENT
# ===========================================================================

elif page == "Predict Segment":
    st.markdown("## 🔍 Predict Customer Segment")
    st.markdown(
        "Enter customer metrics to classify them into a segment "
        "and receive tailored marketing recommendations."
    )

    if not MODEL_LOADED:
        st.error("Model not loaded. Run the training pipeline first.")
        st.stop()

    with st.form("prediction_form"):
        st.markdown("### 📥 Customer Inputs")

        col1, col2, col3 = st.columns(3)
        with col1:
            recency   = st.number_input("Recency (days since last purchase)", min_value=0, max_value=730, value=30)
            tenure    = st.number_input("Tenure (days since first purchase)", min_value=1, max_value=1000, value=180)
        with col2:
            frequency = st.number_input("Frequency (unique orders)", min_value=1, max_value=500, value=10)
            unique_p  = st.number_input("Unique Products Purchased", min_value=1, max_value=2000, value=20)
        with col3:
            monetary  = st.number_input("Monetary (total spend £)", min_value=0.0, max_value=100000.0, value=500.0, step=50.0)
            aov       = st.number_input("Avg Order Value (£)", min_value=0.0, max_value=5000.0, value=50.0, step=5.0)

        st.markdown("#### Optional Advanced Inputs")
        col4, col5, col6 = st.columns(3)
        with col4:
            clv  = st.number_input("Customer Lifetime Value (£)", min_value=0.0, value=0.0, step=100.0, help="Leave 0 to auto-calculate")
        with col5:
            cv   = st.slider("Monthly Spend Variability (CV)", 0.0, 3.0, 0.5, 0.05)
        with col6:
            wsr  = st.slider("Weekend Shopper Ratio", 0.0, 1.0, 0.3, 0.05)

        submitted = st.form_submit_button("🚀 Predict Segment", use_container_width=True)

    if submitted:
        with st.spinner("Scoring customer …"):
            result = predictor.predict_from_rfm(
                recency=recency,
                frequency=frequency,
                monetary=monetary,
                avg_order_value=aov if aov > 0 else None,
                tenure_days=tenure,
                clv=clv if clv > 0 else None,
                unique_products=unique_p,
                monthly_spend_cv=cv,
                weekend_shopper_ratio=wsr,
            )

        st.divider()
        render_prediction(result)

        # Feature gauge chart
        st.markdown("#### 📊 Input Feature Overview")
        feats = {
            "Recency":  recency,
            "Frequency": frequency,
            "Monetary (£)": monetary,
            "AOV (£)": aov,
            "Tenure (days)": tenure,
            "Unique Products": unique_p,
        }
        fig = go.Figure(go.Bar(
            x=list(feats.keys()), y=list(feats.values()),
            marker_color=[result["color"]] * len(feats),
            opacity=0.8,
        ))
        fig.update_layout(
            title="Customer Input Values",
            height=300, showlegend=False,
            yaxis_title="Value",
        )
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE: BATCH SCORING
# ===========================================================================

elif page == "Batch Scoring":
    st.markdown("## 📋 Batch Customer Scoring")
    st.markdown("Upload a CSV with customer RFM data to score all customers at once.")

    if not MODEL_LOADED:
        st.error("Model not loaded. Run the training pipeline first.")
        st.stop()

    # Template download
    template_df = pd.DataFrame({
        "CustomerID":    [12345, 12346, 12347],
        "Recency":       [10, 120, 350],
        "Frequency":     [30, 8, 2],
        "Monetary":      [5200.0, 380.0, 45.0],
        "AvgOrderValue": [173.3, 47.5, 22.5],
        "TenureDays":    [400, 200, 30],
    })
    st.download_button(
        "📥 Download Template CSV",
        template_df.to_csv(index=False),
        "template.csv",
        "text/csv",
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded:
        input_df = pd.read_csv(uploaded)
        st.markdown(f"**Loaded {len(input_df):,} customers**")
        st.dataframe(input_df.head(5), use_container_width=True)

        if st.button("🚀 Run Batch Scoring", use_container_width=True):
            with st.spinner(f"Scoring {len(input_df):,} customers …"):
                result_df = predictor.predict_batch(input_df)

            st.success(f"✅ Scored {len(result_df):,} customers!")
            st.dataframe(result_df, use_container_width=True)

            seg_dist = result_df["Cluster_Name"].value_counts()
            fig = px.bar(
                seg_dist.reset_index(),
                x="Cluster_Name", y="count",
                title="Batch Segment Distribution",
                color="Cluster_Name",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Download results
            st.download_button(
                "📥 Download Scored CSV",
                result_df.to_csv(index=False),
                "scored_customers.csv",
                "text/csv",
            )


# ===========================================================================
# PAGE: SEGMENT INSIGHTS
# ===========================================================================

elif page == "Segment Insights":
    st.markdown("## 💡 Segment Business Insights")
    st.markdown(
        "Deep-dive into each segment's characteristics, "
        "retention risk, and recommended marketing playbooks."
    )

    segments = [
        "Champions", "Loyal Customers", "Potential Loyalists",
        "At Risk", "Hibernating", "Can't Lose Them",
    ]

    selected = st.selectbox("Select Segment", segments)

    profile = CLUSTER_PROFILES.get(selected, CLUSTER_PROFILES["Unknown"])

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{profile['color']}33,{profile['color']}11);
                    border:2px solid {profile['color']}66; border-radius:16px; padding:20px; text-align:center;">
          <div style="font-size:3rem;">{profile['emoji']}</div>
          <h3 style="color:{profile['color']};margin:8px 0">{selected}</h3>
          <p style="color:#aaa;font-size:0.85rem">{profile['rfm_profile']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🔑 Key Metrics")
        st.metric("Retention Risk",   profile["retention_risk"])
        st.metric("Upsell Potential", profile["upsell_potential"])

    with col2:
        st.markdown("#### 📝 Description")
        st.info(profile["description"])

        st.markdown("#### 🎯 Marketing Playbook")
        for i, strategy in enumerate(profile["strategy"], 1):
            st.markdown(
                f'<div class="strategy-item"><b>#{i}</b> {strategy}</div>',
                unsafe_allow_html=True,
            )

    # If clustered data available, show segment stats
    if cluster_data is not None:
        seg_df = cluster_data[cluster_data["Cluster_Name"] == selected]
        if len(seg_df) > 0:
            st.divider()
            st.markdown(f"#### 📊 Live Segment Statistics ({len(seg_df):,} customers)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Customers", f"{len(seg_df):,}")
            c2.metric("Avg Recency", f"{seg_df['Recency'].mean():.0f}d")
            c3.metric("Avg Revenue", f"£{seg_df['Monetary'].mean():,.0f}")
            c4.metric("Avg CLV", f"£{seg_df['CLV'].mean():,.0f}")


# ===========================================================================
# PAGE: METHODOLOGY
# ===========================================================================

elif page == "Methodology":
    st.markdown("## 📚 Methodology & Architecture")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Pipeline", "Dataset", "Algorithms", "Metrics"]
    )

    with tab1:
        st.markdown("""
        ### ML Pipeline Architecture

        ```
        Raw Data (Online Retail II)
             │
             ▼
        ┌─────────────────────────┐
        │   Data Preprocessing    │  ← Remove nulls, duplicates,
        │   (data_preprocessing)  │    cancelled orders, outliers
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │  Feature Engineering    │  ← RFM + CLV + AOV + Tenure
        │  (feature_engineering)  │    + 7 advanced behavioural
        └───────────┬─────────────┘    features per customer
                    │
                    ▼
        ┌─────────────────────────┐
        │   RobustScaler          │  ← Handles outliers better
        │                         │    than StandardScaler
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │   PCA (optional)        │  ← Reduce to 2D for visualisation
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │   Clustering Models     │  ← K-Means (primary)
        │   (train_model)         │    Hierarchical, DBSCAN, GMM
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │   Evaluation & Labels   │  ← Silhouette, Davies-Bouldin,
        │                         │    Calinski-Harabasz
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │   Streamlit App         │  ← Real-time prediction API
        │   (predict.py)          │    + batch scoring
        └─────────────────────────┘
        ```
        """)

    with tab2:
        st.markdown("""
        ### Dataset: UCI Online Retail II

        | Property      | Value                                   |
        |---------------|-----------------------------------------|
        | Source        | Kaggle / UCI ML Repository              |
        | URL           | kaggle.com/datasets/mashlyn/online-retail-ii-uci |
        | Records       | ~1,067,371 transactions                 |
        | Customers     | ~5,878 unique                           |
        | Products      | ~4,631 unique                           |
        | Time Period   | Dec 2009 – Dec 2011 (2 years)           |
        | Geography     | UK-based online retailer (+ 40 countries)|

        #### Business Context
        A UK-based and registered non-store online retail sells unique
        all-occasion giftware. Many customers are wholesalers.

        #### Column Dictionary
        | Column        | Type     | Description                           |
        |---------------|----------|---------------------------------------|
        | InvoiceNo     | String   | 6-digit invoice number (C=cancellation)|
        | StockCode     | String   | Product code                          |
        | Description   | String   | Product name                          |
        | Quantity      | Integer  | Units per transaction                 |
        | InvoiceDate   | DateTime | Date and time of invoice              |
        | UnitPrice     | Float    | Price per unit (£)                    |
        | CustomerID    | Integer  | 5-digit unique customer identifier    |
        | Country       | String   | Customer's country                    |
        """)

    with tab3:
        st.markdown("""
        ### Algorithms

        #### 1. K-Means (Primary — Production Model)
        - **Why**: Fast, interpretable, scales well to thousands of customers
        - **k selection**: Elbow method + Silhouette score
        - **Init**: k-means++ (smarter initialisation, better convergence)
        - **Preprocessing**: RobustScaler (handles outliers in spend data)

        #### 2. Hierarchical Clustering
        - **Why**: No need to specify k; reveals hierarchy in customer data
        - **Linkage**: Ward (minimises within-cluster variance)
        - **Limitation**: O(n²) memory — not suitable for millions of rows

        #### 3. DBSCAN
        - **Why**: Detects arbitrary shapes; marks outliers as noise (-1)
        - **Use case**: Identifying anomalous customers
        - **Limitation**: Sensitive to eps/min_samples tuning

        #### 4. Gaussian Mixture Models
        - **Why**: Soft assignments; captures elliptical cluster shapes
        - **Use case**: Customers on the boundary between segments
        - **Covariance**: Full (each cluster has own covariance matrix)
        """)

    with tab4:
        st.markdown("""
        ### Evaluation Metrics

        | Metric               | Interpretation | Better When |
        |----------------------|----------------|-------------|
        | **Silhouette Score** | How similar objects are to their own cluster vs others. Range: -1 to 1. | Higher ↑ |
        | **Davies-Bouldin**   | Average ratio of within-cluster to between-cluster distances | Lower ↓ |
        | **Calinski-Harabasz**| Ratio of between-cluster to within-cluster dispersion | Higher ↑ |
        | **Inertia (WCSS)**   | Sum of squared distances to nearest centroid (Elbow method) | Lower ↓ |

        #### Why K-Means Won
        K-Means achieved the best Silhouette Score and Calinski-Harabasz
        among all tested algorithms, while producing the most
        interpretable and business-actionable clusters.
        """)

    st.divider()
    st.markdown("""
    **GitHub**: `https://github.com/shubhamkjha369/customer-segmentation`  
    **Built with**: Python 3.10 · scikit-learn · Streamlit · Plotly  
    **Deployment**: Docker → Streamlit Community Cloud / Render
    """)


# ===========================================================================
# FOOTER
# ===========================================================================

st.markdown("---")
st.markdown(
    "<center><small>Customer Segmentation ML System · "
    "Built with scikit-learn & Streamlit · "
    "UCI Online Retail II Dataset</small></center>",
    unsafe_allow_html=True,
)
