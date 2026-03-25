"""
notebooks/01_eda_and_modeling.py
=================================
Complete Exploratory Data Analysis + Modelling walkthrough.

Run this as a Jupyter notebook (jupytext) or plain Python script.
All plots are saved to docs/eda_plots/.

Sections
--------
  §1  Environment setup
  §2  Load & validate raw data
  §3  Data quality audit
  §4  Data cleaning
  §5  Feature engineering (RFM + Advanced)
  §6  EDA — Univariate
  §7  EDA — Bivariate & Correlations
  §8  EDA — Customer behaviour
  §9  Dimensionality reduction (PCA)
  §10 Clustering — model selection
  §11 Cluster profiling & business insights
  §12 Export artefacts
"""

# %% [markdown]
# # 🛍️ Customer Segmentation — End-to-End Analysis
# **Dataset**: UCI Online Retail II (Kaggle)
# **Author**: Your Name
# **Date**: 2024

# %% §1 — Environment setup
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

# ── project path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, get_logger, ensure_dir
from src.data_preprocessing import run_preprocessing
from src.feature_engineering import run_feature_engineering
from src.train_model import run_training

warnings.filterwarnings("ignore")
PLOT_DIR = ensure_dir(PROJECT_ROOT / "docs" / "eda_plots")
sns.set_theme(style="whitegrid", palette="husl")
cfg = load_config()
log = get_logger("EDA", cfg)

print("✅ Environment ready")

# %% §2 — Load raw data
log.info("Loading data …")
clean_df = run_preprocessing(cfg)

print(f"Shape          : {clean_df.shape}")
print(f"Date range     : {clean_df['InvoiceDate'].min().date()} → "
      f"{clean_df['InvoiceDate'].max().date()}")
print(f"Unique customers: {clean_df['CustomerID'].nunique():,}")
print(f"Unique products : {clean_df['StockCode'].nunique():,}")
print(f"Countries       : {clean_df['Country'].nunique()}")
print("\nData types:\n", clean_df.dtypes)

# %% §3 — Data quality audit
print("\n=== MISSING VALUES ===")
print(clean_df.isnull().sum())

print("\n=== DESCRIPTIVE STATISTICS ===")
print(clean_df[["Quantity", "UnitPrice", "TotalRevenue"]].describe().round(2))

# %% §4 — EDA: Revenue distributions

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Revenue-Related Distributions (Log Scale)", fontsize=14, fontweight="bold")

for ax, col, color in zip(
    axes,
    ["Quantity", "UnitPrice", "TotalRevenue"],
    ["#3498DB", "#E74C3C", "#2ECC71"],
):
    data = clean_df[col].dropna()
    data_log = np.log1p(data)
    ax.hist(data_log, bins=50, color=color, alpha=0.8, edgecolor="white")
    ax.set_title(f"log1p({col})")
    ax.set_xlabel(f"log1p({col})")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(PLOT_DIR / "06_revenue_distributions.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Revenue distributions plot saved.")

# %% §4b — Top countries by revenue
top_countries = (
    clean_df.groupby("Country")["TotalRevenue"]
    .sum()
    .sort_values(ascending=False)
    .head(12)
    .reset_index()
)

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.barh(
    top_countries["Country"][::-1],
    top_countries["TotalRevenue"][::-1] / 1e3,
    color=sns.color_palette("husl", 12),
    edgecolor="white",
)
ax.set_title("Top 12 Countries by Total Revenue (£K)", fontsize=13, fontweight="bold")
ax.set_xlabel("Revenue (£ thousands)")
for bar, val in zip(bars, (top_countries["TotalRevenue"][::-1] / 1e3)):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
            f"£{val:,.0f}K", va="center", fontsize=8)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(PLOT_DIR / "07_revenue_by_country.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Country revenue plot saved.")

# %% §4c — Monthly revenue trend
monthly_rev = (
    clean_df.groupby(clean_df["InvoiceDate"].dt.to_period("M"))["TotalRevenue"]
    .sum()
    .reset_index()
)
monthly_rev["InvoiceDate"] = monthly_rev["InvoiceDate"].astype(str)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(monthly_rev["InvoiceDate"], monthly_rev["TotalRevenue"] / 1e3,
        marker="o", linewidth=2, color="#3498DB", markersize=6)
ax.fill_between(
    monthly_rev["InvoiceDate"],
    monthly_rev["TotalRevenue"] / 1e3,
    alpha=0.15, color="#3498DB",
)
ax.set_title("Monthly Revenue Trend", fontsize=13, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue (£ thousands)")
ax.tick_params(axis="x", rotation=45)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(PLOT_DIR / "08_monthly_revenue_trend.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Monthly trend plot saved.")

# %% §4d — Purchase hour & day heatmap
pivot = (
    clean_df.groupby(["DayOfWeek", "Hour"])["TotalRevenue"]
    .sum()
    .reset_index()
    .pivot(index="DayOfWeek", columns="Hour", values="TotalRevenue")
    .fillna(0)
)
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
pivot = pivot.reindex([d for d in day_order if d in pivot.index])

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(
    pivot / 1e3, ax=ax, cmap="YlOrRd",
    linewidths=0.3, annot=False,
    cbar_kws={"label": "Revenue (£K)"},
)
ax.set_title("Revenue Heatmap by Day × Hour", fontsize=13, fontweight="bold")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Day of Week")
plt.tight_layout()
fig.savefig(PLOT_DIR / "09_revenue_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Revenue heatmap saved.")

# %% §5 — Feature Engineering
feat, X_scaled, feat_names = run_feature_engineering(clean_df, cfg)

print(f"\nFeature matrix shape : {feat.shape}")
print(f"Modelling features   : {feat_names}")
print("\nRFM Summary:\n", feat[["Recency", "Frequency", "Monetary"]].describe().round(2))

# %% §6 — RFM Distribution boxplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("RFM Feature Distributions", fontsize=14, fontweight="bold")

for ax, col, color in zip(
    axes, ["Recency", "Frequency", "Monetary"],
    ["#E74C3C", "#3498DB", "#2ECC71"]
):
    data = feat[col].dropna()
    ax.boxplot(data, patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.6),
               medianprops=dict(color="black", linewidth=2))
    ax.set_title(col)
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)
    # Add percentile labels
    for pct, val in zip([25, 50, 75], np.percentile(data, [25, 50, 75])):
        ax.axhline(val, linestyle=":", color="gray", alpha=0.5)

plt.tight_layout()
fig.savefig(PLOT_DIR / "10_rfm_boxplots.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% §7 — Correlation heatmap
num_cols = feat[feat_names].select_dtypes(include=np.number).columns.tolist()
corr_matrix = feat[num_cols].corr(method="spearman").round(2)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(
    corr_matrix, mask=mask, ax=ax,
    annot=True, fmt=".2f", cmap="coolwarm",
    center=0, vmin=-1, vmax=1,
    linewidths=0.4, cbar_kws={"shrink": 0.8},
    annot_kws={"size": 8},
)
ax.set_title("Spearman Correlation Matrix — Customer Features", fontsize=13)
plt.tight_layout()
fig.savefig(PLOT_DIR / "11_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Correlation heatmap saved.")

# %% §8 — RFM Segment distribution
seg_counts = feat["RFM_Segment"].value_counts()

fig, ax = plt.subplots(figsize=(12, 6))
colors = sns.color_palette("husl", len(seg_counts))
bars = ax.bar(seg_counts.index, seg_counts.values, color=colors, edgecolor="white")
ax.set_title("Customer Distribution by RFM Segment", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Customers")
ax.tick_params(axis="x", rotation=30)
for bar, val in zip(bars, seg_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            str(val), ha="center", fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(PLOT_DIR / "12_rfm_segments.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% §9 — PCA Scree plot
pca_full = PCA().fit(X_scaled.values)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("PCA Analysis", fontsize=13, fontweight="bold")

axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
            pca_full.explained_variance_ratio_, color="#3498DB", alpha=0.8)
axes[0].set_title("Explained Variance per Component")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, len(cum_var) + 1), cum_var, "bo-", linewidth=2)
axes[1].axhline(0.95, color="red", linestyle="--", label="95% threshold")
axes[1].set_title("Cumulative Explained Variance")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Explained Variance")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(PLOT_DIR / "13_pca_scree.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("PCA scree plot saved.")

n_95 = int(np.searchsorted(cum_var, 0.95)) + 1
print(f"Components needed for 95% variance: {n_95}")

# %% §10 — Clustering (full pipeline)
feat_labelled, all_labels = run_training(feat, X_scaled, cfg)

# %% §11 — Cluster business insights
print("\n" + "=" * 70)
print("  CLUSTER BUSINESS PROFILES")
print("=" * 70)

cluster_summary = (
    feat_labelled
    .groupby("Cluster_Name")
    .agg(
        Count=("Recency", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Avg_CLV=("CLV", "mean"),
        Avg_AOV=("AvgOrderValue", "mean"),
        Avg_Tenure=("TenureDays", "mean"),
    )
    .round(1)
    .sort_values("Avg_Monetary", ascending=False)
)
print(cluster_summary.to_string())

# Revenue concentration (Pareto analysis)
revenue_by_cluster = (
    feat_labelled.groupby("Cluster_Name")["Monetary"].sum().sort_values(ascending=False)
)
total_rev = revenue_by_cluster.sum()
print(f"\nRevenue Concentration:")
for seg, rev in revenue_by_cluster.items():
    print(f"  {seg:<30}: £{rev:>12,.0f}  ({rev/total_rev:.1%})")

# %% §11b — Cluster profile heatmap
norm_summary = (cluster_summary
                .drop(columns=["Count"])
                .apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)))

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(
    norm_summary, ax=ax, annot=True, fmt=".2f",
    cmap="RdYlGn", linewidths=0.4,
    cbar_kws={"label": "Normalised Score"},
)
ax.set_title("Cluster Feature Profiles (Normalised)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(PLOT_DIR / "14_cluster_profile_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% §12 — Done
print("\n✅ Full EDA + Modelling notebook complete.")
print(f"All plots saved to: {PLOT_DIR}")
