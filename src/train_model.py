"""
train_model.py
==============
Full unsupervised learning pipeline: PCA dimensionality reduction +
four clustering algorithms with comprehensive evaluation metrics.

Algorithms
----------
  1. K-Means          (primary production model)
  2. Agglomerative Hierarchical Clustering
  3. DBSCAN
  4. Gaussian Mixture Models

Evaluation
----------
  - Elbow Method (inertia curve)
  - Silhouette Score
  - Davies–Bouldin Score
  - Calinski–Harabasz Score

Outputs
-------
  models/kmeans_model.pkl
  models/pca_transformer.pkl
  models/feature_names.pkl
  data/processed/rfm_features_clustered.csv
  docs/eda_plots/  (evaluation + cluster plots)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
    silhouette_samples,
)
from sklearn.mixture import GaussianMixture

from src.utils import (
    PROJECT_ROOT,
    ensure_dir,
    get_logger,
    load_config,
    save_artifact,
)

warnings.filterwarnings("ignore")
_log = get_logger(__name__)

PLOT_DIR = PROJECT_ROOT / "docs" / "eda_plots"


# =============================================================================
# HELPER: save figure
# =============================================================================

def _save_fig(fig: plt.Figure, name: str) -> None:
    ensure_dir(PLOT_DIR)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log.info(f"Plot saved: {path.name}")


# =============================================================================
# PCA REDUCER
# =============================================================================

class PCAReducer:
    """Wrap sklearn PCA for reuse in predict pipeline."""

    def __init__(self, n_components: int | float = 2):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.pca.fit_transform(X.values)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.pca.transform(X.values)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_


# =============================================================================
# ELBOW & SILHOUETTE ANALYSIS
# =============================================================================

def elbow_and_silhouette_analysis(
    X: pd.DataFrame, k_range: range, random_state: int = 42
) -> tuple[list[float], list[float], int]:
    """
    Run K-Means for each k in k_range; compute inertia + silhouette.

    Returns
    -------
    inertias, silhouettes, optimal_k
    """
    _log.info(f"Elbow analysis: k = {list(k_range)}")
    inertias, silhouettes = [], []
    X_arr = X.values

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                    max_iter=300, random_state=random_state)
        labels = km.fit_predict(X_arr)
        inertias.append(km.inertia_)

        if k >= 2:
            sil = silhouette_score(X_arr, labels, sample_size=min(5000, len(X_arr)))
            silhouettes.append(sil)
        else:
            silhouettes.append(np.nan)

    # Knee detection (simple: largest drop in inertia gradient)
    deltas = np.diff(inertias)
    second_deriv = np.diff(deltas)
    knee_idx = np.argmax(second_deriv) + 2  # +2 because of double diff offset
    optimal_k = list(k_range)[knee_idx]

    _log.info(
        f"Optimal k by elbow: {optimal_k} | "
        f"Best silhouette: {max(s for s in silhouettes if not np.isnan(s)):.4f}"
    )

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("K Selection: Elbow & Silhouette", fontsize=14, fontweight="bold")

    k_list = list(k_range)
    axes[0].plot(k_list, inertias, "bo-", linewidth=2, markersize=7)
    axes[0].axvline(optimal_k, color="red", linestyle="--", label=f"Knee k={optimal_k}")
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (WCSS)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    valid_k    = [k for k, s in zip(k_list, silhouettes) if not np.isnan(s)]
    valid_sil  = [s for s in silhouettes if not np.isnan(s)]
    best_k_sil = valid_k[np.argmax(valid_sil)]
    axes[1].plot(valid_k, valid_sil, "gs-", linewidth=2, markersize=7)
    axes[1].axvline(best_k_sil, color="red", linestyle="--",
                    label=f"Best k={best_k_sil}")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, "01_elbow_silhouette.png")

    return inertias, silhouettes, optimal_k


# =============================================================================
# SILHOUETTE DIAGRAM
# =============================================================================

def plot_silhouette_diagram(
    X: np.ndarray, labels: np.ndarray, k: int, model_name: str
) -> None:
    """Full silhouette diagram for a specific clustering result."""
    sample_silhouette_values = silhouette_samples(X, labels)
    avg_score = silhouette_score(X, labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10

    cmap = cm.get_cmap("tab10", k)
    for i in range(k):
        ith_cluster_values = np.sort(
            sample_silhouette_values[labels == i]
        )
        size_i = len(ith_cluster_values)
        y_upper = y_lower + size_i

        color = cmap(i)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper), 0, ith_cluster_values,
            facecolor=color, edgecolor=color, alpha=0.7,
        )
        ax.text(-0.05, y_lower + 0.5 * size_i, str(i))
        y_lower = y_upper + 10

    ax.axvline(x=avg_score, color="red", linestyle="--",
               label=f"Avg = {avg_score:.3f}")
    ax.set_title(f"Silhouette Diagram — {model_name} (k={k})", fontsize=13)
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, f"02_silhouette_{model_name.lower().replace(' ', '_')}.png")


# =============================================================================
# CLUSTERING MODELS
# =============================================================================

class ClusteringPipeline:
    """
    Orchestrates fitting, evaluation, and comparison of all clustering models.

    Parameters
    ----------
    config : dict  – project config
    """

    def __init__(self, config: dict):
        self.config = config
        self.cfg_m  = config.get("modeling", {})
        self.logger = get_logger(self.__class__.__name__, config)
        self.results: dict[str, dict] = {}
        self._pca_reducer: PCAReducer | None = None

    # ------------------------------------------------------------------

    def fit_all(
        self, X: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """
        Fit all algorithms, evaluate, and return the best labels.

        Parameters
        ----------
        X : pd.DataFrame – scaled feature matrix (CustomerID as index)

        Returns
        -------
        evaluation_df : comparison table
        all_labels    : {model_name: label_array}
        """
        self.logger.info("=" * 60)
        self.logger.info("  CLUSTERING PIPELINE — ALL MODELS")
        self.logger.info("=" * 60)

        # ── PCA for 2-D visualisation ─────────────────────────────────
        pca_cfg = self.cfg_m.get("pca", {})
        self._pca_reducer = PCAReducer(n_components=2)
        X_2d = self._pca_reducer.fit_transform(X)
        self.logger.info(
            f"PCA 2D: explained variance = "
            f"{self._pca_reducer.explained_variance_ratio.sum():.2%}"
        )

        all_labels: dict[str, np.ndarray] = {}

        # ── 1. Elbow & Silhouette analysis ────────────────────────────
        k_cfg   = self.cfg_m.get("kmeans", {})
        k_range = range(*k_cfg.get("k_range", [2, 11]))
        _, _, optimal_k = elbow_and_silhouette_analysis(X, k_range)

        # Override with config value if explicitly set
        k = k_cfg.get("optimal_k", optimal_k)
        self.logger.info(f"Using k = {k} for K-Means (and comparable models)")

        # ── 2. K-Means ────────────────────────────────────────────────
        km_labels, km_model = self._fit_kmeans(X, k)
        all_labels["K-Means"] = km_labels
        self.results["K-Means"] = self._evaluate(X.values, km_labels, "K-Means")

        # Save the primary production model
        save_artifact(km_model, self.config["model"]["kmeans_filename"], self.config)
        self.logger.info("K-Means model saved.")

        # ── 3. Hierarchical ───────────────────────────────────────────
        hc_cfg = self.cfg_m.get("hierarchical", {})
        hc_labels = self._fit_hierarchical(X, k)
        all_labels["Hierarchical"] = hc_labels
        self.results["Hierarchical"] = self._evaluate(X.values, hc_labels, "Hierarchical")

        # ── 4. DBSCAN ─────────────────────────────────────────────────
        db_cfg = self.cfg_m.get("dbscan", {})
        db_labels = self._fit_dbscan(X, db_cfg)
        all_labels["DBSCAN"] = db_labels
        if len(np.unique(db_labels[db_labels >= 0])) >= 2:
            # Only evaluate non-noise points
            mask = db_labels >= 0
            self.results["DBSCAN"] = self._evaluate(
                X.values[mask], db_labels[mask], "DBSCAN"
            )
        else:
            self.results["DBSCAN"] = {
                "Silhouette": np.nan,
                "Davies-Bouldin": np.nan,
                "Calinski-Harabasz": np.nan,
                "n_clusters": len(np.unique(db_labels[db_labels >= 0])),
            }

        # ── 5. Gaussian Mixture Model ─────────────────────────────────
        gmm_cfg = self.cfg_m.get("gmm", {})
        gmm_labels, gmm_model = self._fit_gmm(X, k, gmm_cfg)
        all_labels["GMM"] = gmm_labels
        self.results["GMM"] = self._evaluate(X.values, gmm_labels, "GMM")

        # ── Comparison table ──────────────────────────────────────────
        eval_df = pd.DataFrame(self.results).T
        eval_df = eval_df.round(4)
        self.logger.info(f"\nModel Comparison:\n{eval_df.to_string()}")

        # ── Visualisations ────────────────────────────────────────────
        self._plot_pca_clusters(X_2d, all_labels)
        self._plot_model_comparison(eval_df)
        plot_silhouette_diagram(X.values, km_labels, k, "K-Means")
        self._plot_cluster_profiles(X, km_labels)

        # Save PCA reducer
        save_artifact(
            self._pca_reducer,
            self.config["model"]["pca_filename"],
            self.config,
        )

        return eval_df, all_labels

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def _fit_kmeans(
        self, X: pd.DataFrame, k: int
    ) -> tuple[np.ndarray, KMeans]:
        k_cfg = self.cfg_m.get("kmeans", {})
        model = KMeans(
            n_clusters=k,
            init=k_cfg.get("init", "k-means++"),
            n_init=k_cfg.get("n_init", 10),
            max_iter=k_cfg.get("max_iter", 300),
            random_state=self.cfg_m.get("random_state", 42),
        )
        labels = model.fit_predict(X.values)
        self.logger.info(
            f"K-Means fit: k={k} | "
            f"inertia={model.inertia_:.2f} | "
            f"clusters={np.bincount(labels).tolist()}"
        )
        return labels, model

    def _fit_hierarchical(self, X: pd.DataFrame, k: int) -> np.ndarray:
        hc_cfg = self.cfg_m.get("hierarchical", {})
        model  = AgglomerativeClustering(
            n_clusters=k,
            linkage=hc_cfg.get("linkage", "ward"),
        )
        labels = model.fit_predict(X.values)
        self.logger.info(
            f"Hierarchical fit: k={k} | "
            f"clusters={np.bincount(labels).tolist()}"
        )
        return labels

    def _fit_dbscan(self, X: pd.DataFrame, db_cfg: dict) -> np.ndarray:
        model = DBSCAN(
            eps=db_cfg.get("eps", 0.5),
            min_samples=db_cfg.get("min_samples", 5),
            metric=db_cfg.get("metric", "euclidean"),
        )
        labels = model.fit_predict(X.values)
        n_clusters = len(np.unique(labels[labels >= 0]))
        n_noise    = (labels == -1).sum()
        self.logger.info(
            f"DBSCAN fit: {n_clusters} clusters | {n_noise} noise points"
        )
        return labels

    def _fit_gmm(
        self, X: pd.DataFrame, k: int, gmm_cfg: dict
    ) -> tuple[np.ndarray, GaussianMixture]:
        model = GaussianMixture(
            n_components=k,
            covariance_type=gmm_cfg.get("covariance_type", "full"),
            max_iter=gmm_cfg.get("max_iter", 200),
            random_state=self.cfg_m.get("random_state", 42),
        )
        labels = model.fit_predict(X.values)
        self.logger.info(
            f"GMM fit: k={k} | "
            f"BIC={model.bic(X.values):.2f} | "
            f"clusters={np.bincount(labels).tolist()}"
        )
        return labels, model

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate(X: np.ndarray, labels: np.ndarray, name: str) -> dict:
        n_labels = len(np.unique(labels))
        if n_labels < 2:
            return {
                "Silhouette": np.nan,
                "Davies-Bouldin": np.nan,
                "Calinski-Harabasz": np.nan,
                "n_clusters": n_labels,
            }
        return {
            "Silhouette":          round(silhouette_score(X, labels, sample_size=min(5000, len(X))), 4),
            "Davies-Bouldin":      round(davies_bouldin_score(X, labels), 4),
            "Calinski-Harabasz":   round(calinski_harabasz_score(X, labels), 4),
            "n_clusters":          n_labels,
        }

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _plot_pca_clusters(
        self,
        X_2d: np.ndarray,
        all_labels: dict[str, np.ndarray],
    ) -> None:
        """2-D PCA scatter for each algorithm, side by side."""
        n = len(all_labels)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]
        fig.suptitle("Cluster Visualisation (PCA 2-D)", fontsize=14, fontweight="bold")

        cmap = "tab10"
        for ax, (name, labels) in zip(axes, all_labels.items()):
            scatter = ax.scatter(
                X_2d[:, 0], X_2d[:, 1],
                c=labels, cmap=cmap,
                s=8, alpha=0.5,
            )
            ax.set_title(name, fontsize=12)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            plt.colorbar(scatter, ax=ax, label="Cluster")

        plt.tight_layout()
        _save_fig(fig, "03_pca_clusters.png")

    def _plot_model_comparison(self, eval_df: pd.DataFrame) -> None:
        """Bar chart comparing evaluation metrics across models."""
        metrics = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]
        available = [m for m in metrics if m in eval_df.columns]

        fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
        if len(available) == 1:
            axes = [axes]
        fig.suptitle("Model Evaluation Comparison", fontsize=14, fontweight="bold")

        colours = sns.color_palette("husl", len(eval_df))
        for ax, metric in zip(axes, available):
            vals = eval_df[metric].dropna().astype(float)
            bars = ax.bar(vals.index, vals.values, color=colours[:len(vals)], edgecolor="black")
            ax.set_title(metric)
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=15)
            for bar, val in zip(bars, vals.values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:.3f}", ha="center", fontsize=9,
                )
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        _save_fig(fig, "04_model_comparison.png")

    def _plot_cluster_profiles(
        self, X: pd.DataFrame, labels: np.ndarray
    ) -> None:
        """Radar / snake plot of normalised feature means per cluster."""
        df = X.copy()
        df["Cluster"] = labels

        # Select top numeric features for readability
        top_features = ["Recency", "Frequency", "Monetary",
                        "AvgOrderValue", "CLV", "TenureDays",
                        "PurchaseFreqMonthly", "UniqueProducts"]
        available = [f for f in top_features if f in df.columns]

        means = df.groupby("Cluster")[available].mean()
        # Normalise 0–1 for comparison
        norm  = (means - means.min()) / (means.max() - means.min() + 1e-9)

        fig, ax = plt.subplots(figsize=(12, 6))
        cmap = plt.get_cmap("tab10", len(means))
        for i, row in norm.iterrows():
            ax.plot(available, row.values, marker="o",
                    label=f"Cluster {i}", linewidth=2, color=cmap(i))
            ax.fill_between(range(len(available)), row.values, alpha=0.08,
                            color=cmap(i))

        ax.set_title("Normalised Cluster Feature Profiles (K-Means)", fontsize=14)
        ax.set_xticks(range(len(available)))
        ax.set_xticklabels(available, rotation=30, ha="right")
        ax.set_ylabel("Normalised Mean Value")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save_fig(fig, "05_cluster_profiles.png")


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_training(
    feat: pd.DataFrame,
    X_scaled: pd.DataFrame,
    config: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Execute the full clustering pipeline and attach labels to feat.

    Returns
    -------
    feat_with_labels : pd.DataFrame
    all_labels       : dict[model_name, np.ndarray]
    """
    if config is None:
        config = load_config()

    pipeline = ClusteringPipeline(config)
    eval_df, all_labels = pipeline.fit_all(X_scaled)

    # Attach best-model (K-Means) labels to feature matrix
    feat = feat.copy()
    feat["Cluster"] = all_labels["K-Means"]

    # Save cluster labels
    cluster_labels = config.get("cluster_labels", {})
    feat["Cluster_Name"] = feat["Cluster"].map(
        {int(k): v for k, v in cluster_labels.items()}
    ).fillna("Unknown")

    # Save enriched features
    out_path = PROJECT_ROOT / "data" / "processed" / "rfm_features_clustered.csv"
    feat.to_csv(out_path)
    _log.info(f"Clustered feature matrix saved → {out_path.name}")

    # Save feature names list
    import pickle
    feat_names_path = (
        PROJECT_ROOT
        / config.get("model", {}).get("save_dir", "models")
        / config.get("model", {}).get("features_filename", "feature_names.pkl")
    )
    feat_names_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feat_names_path, "wb") as f:
        pickle.dump(list(X_scaled.columns), f)

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(eval_df.to_string())
    print("=" * 60 + "\n")

    return feat, all_labels


if __name__ == "__main__":
    from src.data_preprocessing import run_preprocessing
    from src.feature_engineering import run_feature_engineering

    cfg   = load_config()
    clean = run_preprocessing(cfg)
    feat, X, names = run_feature_engineering(clean, cfg)
    feat_labelled, all_labels = run_training(feat, X, cfg)
    print(feat_labelled[["Recency", "Frequency", "Monetary",
                          "Cluster", "Cluster_Name"]].head(10))
