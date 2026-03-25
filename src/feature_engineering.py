"""
feature_engineering.py
======================
RFM (Recency, Frequency, Monetary) feature engineering plus additional
behavioural features to enrich the customer representation.

Features Created
----------------
Core RFM:
  Recency          – Days since last purchase
  Frequency        – Number of unique invoices
  Monetary         – Total spend

Advanced:
  CLV              – Customer Lifetime Value (proxy: Monetary × Frequency / Tenure)
  AvgOrderValue    – Monetary / Frequency
  PurchaseFreq     – Invoices per active month
  UniqueProd       – Count of distinct products purchased
  AvgBasketSize    – Average items per invoice
  ReturnsRate      – Not applicable after cancellation removal
  TenureDays       – Days from first to last purchase
  MonthlySpendCV   – Coefficient of variation in monthly spend (consistency)
  FavouriteDay     – Most frequent purchase day of week
  FavouriteHour    – Most frequent purchase hour bucket

Scoring:
  R_Score, F_Score, M_Score – Quartile-based scores (1–4)
  RFM_Score                 – Sum of individual scores
  RFM_Segment               – Named segment label (Snake → Champion → Hibernating …)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.utils import (
    PROJECT_ROOT,
    get_logger,
    load_config,
    resolve_path,
    save_artifact,
    validate_rfm_dataframe,
)

warnings.filterwarnings("ignore")

_log = get_logger(__name__)


# =============================================================================
# RFM FEATURE BUILDER
# =============================================================================

class RFMFeatureBuilder:
    """
    Build an RFM + advanced-feature matrix from cleaned transaction data.

    Parameters
    ----------
    config : dict  – full project config
    """

    # RFM Segment map (score → label) – classic 11-segment RFM
    _SEGMENT_MAP = {
        r"[3-4][3-4]": "Champions",
        r"[2-3][3-4]": "Loyal Customers",
        r"[3-4][1-2]": "Potential Loyalists",
        r"4[0-2]":     "Recent Customers",
        r"3[0-2]":     "Promising",
        r"2[1-2]":     "Customers Needing Attention",
        r"[0-1][1-3]": "At Risk",
        r"[0-1][4]":   "Can't Lose Them",
        r"1[1-2]":     "Hibernating",
        r"[0-1][0-1]": "Lost",
    }

    def __init__(self, config: dict):
        self.config  = config
        self.cfg_fe  = config.get("feature_engineering", {})
        self.logger  = get_logger(self.__class__.__name__, config)
        self._scaler = RobustScaler()

    # ------------------------------------------------------------------

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform cleaned transactions → customer-level feature matrix.

        Parameters
        ----------
        df : pd.DataFrame – cleaned transactions (output of DataCleaner)

        Returns
        -------
        pd.DataFrame – one row per customer, columns = features
        """
        self.logger.info("=" * 60)
        self.logger.info("  FEATURE ENGINEERING PIPELINE")
        self.logger.info("=" * 60)

        snapshot = self._snapshot_date(df)
        self.logger.info(f"Snapshot date: {snapshot.date()}")

        rfm  = self._compute_rfm(df, snapshot)
        adv  = self._compute_advanced_features(df, snapshot)
        feat = rfm.join(adv, on="CustomerID", how="left")

        feat = self._add_rfm_scores(feat)
        feat = self._add_rfm_segments(feat)
        feat = self._impute_missing(feat)

        self.logger.info(
            f"Feature matrix: {feat.shape[0]:,} customers × {feat.shape[1]} features"
        )
        return feat

    # ------------------------------------------------------------------
    # Core RFM
    # ------------------------------------------------------------------

    def _snapshot_date(self, df: pd.DataFrame) -> pd.Timestamp:
        offset = self.cfg_fe.get("snapshot_days_offset", 1)
        return df["InvoiceDate"].max() + pd.Timedelta(days=offset)

    def _compute_rfm(
        self, df: pd.DataFrame, snapshot: pd.Timestamp
    ) -> pd.DataFrame:
        """Compute Recency, Frequency, Monetary per customer."""
        rfm = (
            df.groupby("CustomerID")
            .agg(
                LastPurchaseDate=("InvoiceDate", "max"),
                Frequency=("InvoiceNo", "nunique"),
                Monetary=("TotalRevenue", "sum"),
            )
            .reset_index()
        )
        rfm["Recency"] = (snapshot - rfm["LastPurchaseDate"]).dt.days
        rfm = rfm.drop(columns=["LastPurchaseDate"])

        self.logger.info(
            f"RFM computed for {len(rfm):,} customers | "
            f"Recency μ={rfm['Recency'].mean():.1f}d | "
            f"Frequency μ={rfm['Frequency'].mean():.1f} | "
            f"Monetary μ=£{rfm['Monetary'].mean():.2f}"
        )
        return rfm.set_index("CustomerID")

    # ------------------------------------------------------------------
    # Advanced Features
    # ------------------------------------------------------------------

    def _compute_advanced_features(
        self, df: pd.DataFrame, snapshot: pd.Timestamp
    ) -> pd.DataFrame:
        """Compute behavioural metrics beyond RFM."""

        # ── per-customer base stats ────────────────────────────────────
        base = df.groupby("CustomerID").agg(
            FirstPurchaseDate=("InvoiceDate", "min"),
            LastPurchaseDate=("InvoiceDate", "max"),
            TotalRevenue=("TotalRevenue", "sum"),
            TotalOrders=("InvoiceNo", "nunique"),
            TotalItems=("Quantity", "sum"),
            UniqueProducts=("StockCode", "nunique"),
        )

        # Tenure: days between first and last purchase
        base["TenureDays"] = (
            base["LastPurchaseDate"] - base["FirstPurchaseDate"]
        ).dt.days
        base["TenureDays"] = base["TenureDays"].clip(lower=1)

        # Average Order Value
        base["AvgOrderValue"] = base["TotalRevenue"] / base["TotalOrders"]

        # Average Basket Size (items per order)
        base["AvgBasketSize"] = base["TotalItems"] / base["TotalOrders"]

        # Purchase Frequency per month
        base["ActiveMonths"] = (
            base["TenureDays"] / 30.44
        ).clip(lower=1)
        base["PurchaseFreqMonthly"] = base["TotalOrders"] / base["ActiveMonths"]

        # Customer Lifetime Value (simplified proxy)
        base["CLV"] = (
            base["AvgOrderValue"] * base["PurchaseFreqMonthly"] * 12
        )

        # ── Monthly spend variability ──────────────────────────────────
        monthly = (
            df.groupby(["CustomerID", df["InvoiceDate"].dt.to_period("M")])[
                "TotalRevenue"
            ]
            .sum()
            .reset_index()
        )
        monthly.columns = ["CustomerID", "Period", "MonthlySpend"]
        cv = monthly.groupby("CustomerID")["MonthlySpend"].agg(
            lambda x: (x.std() / x.mean()) if x.mean() > 0 else 0
        )
        base["MonthlySpendCV"] = cv

        # ── Favourite day of week & hour ──────────────────────────────
        fav_day = (
            df.groupby("CustomerID")["DayOfWeek"]
            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "Unknown")
        )
        base["FavouriteDay"] = fav_day

        fav_hour_bin = df.copy()
        fav_hour_bin["HourBin"] = pd.cut(
            fav_hour_bin["Hour"],
            bins=[0, 6, 12, 18, 24],
            labels=["Night", "Morning", "Afternoon", "Evening"],
            right=False,
        )
        fav_hour = (
            fav_hour_bin.groupby("CustomerID")["HourBin"]
            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "Unknown")
        )
        base["FavouriteHourBin"] = fav_hour

        # ── Weekend shopper flag ──────────────────────────────────────
        weekend = df.groupby("CustomerID")["IsWeekend"].mean()
        base["WeekendShopperRatio"] = weekend

        # ── Country ──────────────────────────────────────────────────
        country = df.groupby("CustomerID")["Country"].agg(
            lambda x: x.mode().iloc[0]
        )
        base["Country"] = country

        # Drop helper columns
        base = base.drop(
            columns=["FirstPurchaseDate", "LastPurchaseDate", "ActiveMonths"],
            errors="ignore",
        )

        self.logger.info("Advanced features computed.")
        return base

    # ------------------------------------------------------------------
    # RFM Scoring
    # ------------------------------------------------------------------

    def _add_rfm_scores(self, feat: pd.DataFrame) -> pd.DataFrame:
        """
        Assign quartile-based R, F, M scores (1 = worst, 4 = best).

        Recency: lower days = better → reversed quartile
        Frequency & Monetary: higher = better → standard quartile

        Uses a robust fallback to rank-based percentile scoring when
        pd.qcut cannot form the requested number of unique bins (common
        with small or heavily duplicated datasets).
        """
        q = self.cfg_fe.get("rfm_quantiles", 4)

        def _safe_qcut(series: pd.Series, ascending: bool = True) -> pd.Series:
            """
            Try pd.qcut; fall back to rank-based percentile binning if
            duplicate bin edges prevent the requested number of labels.
            """
            n_unique = series.nunique()
            actual_q = min(q, n_unique)          # can't have more bins than unique values
            labels_asc  = list(range(1, actual_q + 1))
            labels_desc = list(range(actual_q, 0, -1))
            labels = labels_asc if ascending else labels_desc

            try:
                result = pd.qcut(
                    series, q=actual_q, labels=labels, duplicates="drop"
                )
            except ValueError:
                # Final fallback: rank-based, always works
                pct = series.rank(pct=True)
                bins = np.linspace(0, 1, actual_q + 1)
                result = pd.cut(
                    pct, bins=bins, labels=labels,
                    include_lowest=True, duplicates="drop"
                )

            # Forward-fill any NaN from edge cases, then cast to int
            result = result.cat.codes.replace(-1, 0)
            # Re-map codes (0..actual_q-1) to labels (1..actual_q)
            code_to_score = {i: labels[i] for i in range(len(labels))}
            result = result.map(code_to_score).fillna(1).astype(int)

            # If actual_q < q, stretch the range back to [1, q]
            if actual_q < q:
                lo, hi = result.min(), result.max()
                if hi > lo:
                    result = (
                        ((result - lo) / (hi - lo) * (q - 1) + 1)
                        .round()
                        .astype(int)
                        .clip(1, q)
                    )
            return result

        feat["R_Score"] = _safe_qcut(feat["Recency"],   ascending=False)  # low recency → high score
        feat["F_Score"] = _safe_qcut(feat["Frequency"], ascending=True)
        feat["M_Score"] = _safe_qcut(feat["Monetary"],  ascending=True)

        feat["RFM_Score"] = feat["R_Score"] + feat["F_Score"] + feat["M_Score"]
        feat["RF_Score"]  = feat["R_Score"].astype(str) + feat["F_Score"].astype(str)

        self.logger.info("RFM quartile scores (R/F/M 1–4) added.")
        return feat

    def _add_rfm_segments(self, feat: pd.DataFrame) -> pd.DataFrame:
        """Map RF_Score to named RFM segment label."""
        import re

        def _map_segment(rf: str) -> str:
            for pattern, label in self._SEGMENT_MAP.items():
                if re.match(pattern, rf):
                    return label
            return "Other"

        feat["RFM_Segment"] = feat["RF_Score"].apply(_map_segment)
        seg_dist = feat["RFM_Segment"].value_counts()
        self.logger.info(f"RFM Segments:\n{seg_dist.to_string()}")
        return feat

    # ------------------------------------------------------------------
    # Imputation
    # ------------------------------------------------------------------

    def _impute_missing(self, feat: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN in numeric columns with the column median."""
        num_cols = feat.select_dtypes(include=np.number).columns
        for col in num_cols:
            n_missing = feat[col].isna().sum()
            if n_missing > 0:
                feat[col] = feat[col].fillna(feat[col].median())
                self.logger.debug(f"Imputed {n_missing} NaN in {col}")
        return feat

    # ------------------------------------------------------------------
    # Scale & Return Model-Ready Feature Matrix
    # ------------------------------------------------------------------

    def get_model_features(
        self, feat: pd.DataFrame, scale: bool = True
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Return a scaled numeric feature matrix ready for clustering.

        Parameters
        ----------
        feat  : pd.DataFrame – output of .build()
        scale : bool         – apply RobustScaler (default True)

        Returns
        -------
        (pd.DataFrame of scaled features, list of feature names)
        """
        # Select only numeric modelling features (exclude scores/labels)
        exclude = {
            "R_Score", "F_Score", "M_Score", "RFM_Score",
            "RF_Score", "RFM_Segment",
            "FavouriteDay", "FavouriteHourBin", "Country",
            "TotalRevenue", "TotalOrders", "TotalItems",  # redundant
        }
        num_cols = [
            c for c in feat.select_dtypes(include=np.number).columns
            if c not in exclude
        ]

        X = feat[num_cols].copy()

        if scale:
            X_scaled = self._scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, index=X.index, columns=num_cols)
            self.logger.info(f"Features scaled with RobustScaler: {num_cols}")

        return X, num_cols

    def save_scaler(self, config: dict) -> None:
        """Persist the fitted scaler."""
        fname = config.get("model", {}).get("scaler_filename", "scaler.pkl")
        save_artifact(self._scaler, fname, config)
        self.logger.info(f"Scaler saved: {fname}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_feature_engineering(
    df: pd.DataFrame, config: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    df     : pd.DataFrame – cleaned transactions
    config : dict         – project config

    Returns
    -------
    feat       : pd.DataFrame – full feature + score + segment matrix
    X_scaled   : pd.DataFrame – model-ready scaled feature matrix
    feat_names : list[str]    – feature column names
    """
    if config is None:
        config = load_config()

    builder   = RFMFeatureBuilder(config)
    feat      = builder.build(df)

    X_scaled, feat_names = builder.get_model_features(feat, scale=True)
    builder.save_scaler(config)

    # Save full feature matrix
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    feat.to_csv(processed_dir / "rfm_features.csv")
    X_scaled.to_csv(processed_dir / "rfm_scaled.csv")

    _log.info(f"Feature engineering done | Shape: {feat.shape}")
    return feat, X_scaled, feat_names


if __name__ == "__main__":
    from src.data_preprocessing import run_preprocessing
    cfg    = load_config()
    clean  = run_preprocessing(cfg)
    feat, X, names = run_feature_engineering(clean, cfg)
    print(feat.head(3))
    print(f"\nModelling features ({len(names)}): {names}")
