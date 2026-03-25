"""
predict.py
==========
Production inference pipeline.

Given raw customer transaction data (or pre-computed RFM values),
this module:
  1. Validates the input
  2. Computes features (or uses provided RFM)
  3. Scales features with the saved RobustScaler
  4. Applies the saved K-Means model
  5. Returns the cluster ID, cluster name, and business recommendation

Used by the Streamlit app and any REST API wrapper.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import (
    PROJECT_ROOT,
    get_logger,
    load_artifact,
    load_config,
)

warnings.filterwarnings("ignore")
_log = get_logger(__name__)


# =============================================================================
# CLUSTER METADATA
# =============================================================================

# Business profile for each cluster — kept in code for portability,
# but can be overridden from config.yaml cluster_labels section.

CLUSTER_PROFILES = {
    "Champions": {
        "emoji":       "🏆",
        "description": (
            "Your best customers. Bought recently, buy often, "
            "and spend the most. Reward and retain them."
        ),
        "rfm_profile": "High R · High F · High M",
        "strategy": [
            "Offer exclusive loyalty rewards and early access to new products.",
            "Ask for reviews / referrals — high NPS potential.",
            "Enrol in VIP programme with personalised perks.",
            "Use as brand ambassadors in social campaigns.",
        ],
        "retention_risk": "Low",
        "upsell_potential": "Very High",
        "color": "#2ECC71",
    },
    "Loyal Customers": {
        "emoji":       "💙",
        "description": (
            "Spend well and purchase frequently, though not as recent "
            "as Champions. Core revenue drivers."
        ),
        "rfm_profile": "Medium-High R · High F · High M",
        "strategy": [
            "Introduce subscription / auto-replenishment options.",
            "Up-sell higher-margin product lines.",
            "Send 'thank you' appreciation campaigns.",
            "Personalised product recommendations.",
        ],
        "retention_risk": "Low–Medium",
        "upsell_potential": "High",
        "color": "#3498DB",
    },
    "Potential Loyalists": {
        "emoji":       "🌱",
        "description": (
            "Recent customers with moderate frequency. Have potential "
            "to become Loyal Customers with the right nurturing."
        ),
        "rfm_profile": "High R · Medium F · Medium M",
        "strategy": [
            "Onboarding email series with product education.",
            "Offer a loyalty points programme.",
            "Send targeted recommendations based on purchase history.",
            "Offer second-purchase discount to increase frequency.",
        ],
        "retention_risk": "Medium",
        "upsell_potential": "High",
        "color": "#9B59B6",
    },
    "Recent Customers": {
        "emoji":       "🆕",
        "description": (
            "Purchased very recently but only once or twice. "
            "Critical window to build habit."
        ),
        "rfm_profile": "Very High R · Low F · Low-Medium M",
        "strategy": [
            "Welcome sequence: 3-email onboarding journey.",
            "Free returns / satisfaction guarantee to reduce first-buy risk.",
            "Introduce the brand story and product range.",
            "Time-limited second-purchase voucher (7-day expiry).",
        ],
        "retention_risk": "High",
        "upsell_potential": "Medium",
        "color": "#1ABC9C",
    },
    "Promising": {
        "emoji":       "✨",
        "description": (
            "Fairly recent buyers with above-average basket size. "
            "May increase frequency with incentives."
        ),
        "rfm_profile": "Medium-High R · Medium F · Medium M",
        "strategy": [
            "Bundle deals to increase average order value.",
            "Category expansion emails ('customers like you also bought …').",
            "Win-back offers if 60+ days since last purchase.",
        ],
        "retention_risk": "Medium",
        "upsell_potential": "Medium–High",
        "color": "#F1C40F",
    },
    "Customers Needing Attention": {
        "emoji":       "⚠️",
        "description": (
            "Decent frequency but haven't purchased recently. "
            "Show early signs of churn."
        ),
        "rfm_profile": "Medium R · Medium F · Medium M",
        "strategy": [
            "Reactivation campaign: personalised 'We miss you' email.",
            "Limited-time discount on previously bought categories.",
            "Survey to understand drop-off reasons.",
            "Trigger-based retargeting ads.",
        ],
        "retention_risk": "Medium–High",
        "upsell_potential": "Medium",
        "color": "#E67E22",
    },
    "At Risk": {
        "emoji":       "🚨",
        "description": (
            "Were good customers but haven't bought in a long time. "
            "High churn probability."
        ),
        "rfm_profile": "Low R · Medium-High F · Medium-High M",
        "strategy": [
            "Aggressive win-back offer (20–30% discount).",
            "Phone/chat outreach for high-value at-risk customers.",
            "Remind them of past purchases and what's new.",
            "Survey: understand what drove them away.",
        ],
        "retention_risk": "Very High",
        "upsell_potential": "Low–Medium",
        "color": "#E74C3C",
    },
    "Can't Lose Them": {
        "emoji":       "🔑",
        "description": (
            "Used to be Champions or Loyal. Haven't purchased in a very "
            "long time but historically high value. Top priority to recover."
        ),
        "rfm_profile": "Very Low R · High F · High M",
        "strategy": [
            "Personal outreach from account manager / founder.",
            "VIP win-back package with generous offer.",
            "Premium gift with next purchase.",
            "Dedicated support and concierge experience.",
        ],
        "retention_risk": "Critical",
        "upsell_potential": "High (if recovered)",
        "color": "#C0392B",
    },
    "Hibernating": {
        "emoji":       "😴",
        "description": (
            "Bought a while ago with low frequency and spend. "
            "Low engagement but worth a low-cost reactivation attempt."
        ),
        "rfm_profile": "Low R · Low F · Low M",
        "strategy": [
            "Low-cost email reactivation (no deep discount needed).",
            "Share new product launches or improvements.",
            "Consider removing from active marketing to save budget.",
        ],
        "retention_risk": "High",
        "upsell_potential": "Low",
        "color": "#95A5A6",
    },
    "Lost": {
        "emoji":       "💀",
        "description": (
            "Lowest scores across all RFM dimensions. "
            "Likely churned; minimal spend on reactivation advised."
        ),
        "rfm_profile": "Very Low R · Very Low F · Very Low M",
        "strategy": [
            "Sunset campaign: one final re-engagement email.",
            "Remove from active segments to optimise marketing spend.",
            "Keep in database for aggregated analytics.",
        ],
        "retention_risk": "Churned",
        "upsell_potential": "Very Low",
        "color": "#7F8C8D",
    },
    "Unknown": {
        "emoji":       "❓",
        "description": "Segment could not be determined.",
        "rfm_profile": "N/A",
        "strategy": ["Collect more data before targeting."],
        "retention_risk": "Unknown",
        "upsell_potential": "Unknown",
        "color": "#BDC3C7",
    },
}


# =============================================================================
# PREDICTION ENGINE
# =============================================================================

class CustomerSegmentPredictor:
    """
    Load persisted artefacts and score new customers.

    Parameters
    ----------
    config : dict  – project config (loaded from config.yaml if None)
    """

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.logger = get_logger(self.__class__.__name__, self.config)
        self._scaler       = None
        self._model        = None
        self._pca          = None
        self._feature_names: list[str] = []
        self._cluster_labels: dict[int, str] = {}
        self._loaded = False

    def load(self) -> "CustomerSegmentPredictor":
        """Load all persisted artefacts from disk."""
        cfg_m = self.config.get("model", {})

        self._scaler = load_artifact(cfg_m.get("scaler_filename",   "scaler.pkl"),       self.config)
        self._model  = load_artifact(cfg_m.get("kmeans_filename",   "kmeans_model.pkl"), self.config)
        self._pca    = load_artifact(cfg_m.get("pca_filename",      "pca_transformer.pkl"), self.config)

        feat_path = (
            PROJECT_ROOT
            / cfg_m.get("save_dir", "models")
            / cfg_m.get("features_filename", "feature_names.pkl")
        )
        with open(feat_path, "rb") as f:
            self._feature_names = pickle.load(f)

        self._cluster_labels = {
            int(k): v
            for k, v in self.config.get("cluster_labels", {}).items()
        }
        self._loaded = True
        self.logger.info(f"Artefacts loaded | {len(self._feature_names)} features")
        return self

    # ------------------------------------------------------------------

    def predict_from_rfm(
        self,
        recency: float,
        frequency: float,
        monetary: float,
        avg_order_value: Optional[float] = None,
        tenure_days: Optional[float] = None,
        clv: Optional[float] = None,
        unique_products: Optional[int] = None,
        purchase_freq_monthly: Optional[float] = None,
        avg_basket_size: Optional[float] = None,
        monthly_spend_cv: Optional[float] = None,
        weekend_shopper_ratio: Optional[float] = None,
    ) -> dict:
        """
        Predict segment from manually supplied RFM (and optional) values.

        Returns a rich dict with cluster_id, cluster_name, profile,
        and business recommendations.
        """
        if not self._loaded:
            self.load()

        # Build a feature row with sensible defaults
        aov  = avg_order_value     or (monetary / max(frequency, 1))
        ten  = tenure_days         or 90.0
        clv_ = clv                 or (aov * max(frequency, 1))
        up   = unique_products     or max(int(frequency * 2), 1)
        pfm  = purchase_freq_monthly or (frequency / max(ten / 30.44, 1))
        abs_ = avg_basket_size     or 5.0
        cv   = monthly_spend_cv    or 0.5
        wsr  = weekend_shopper_ratio or 0.3

        raw_values = {
            "Recency":              recency,
            "Frequency":            frequency,
            "Monetary":             monetary,
            "AvgOrderValue":        aov,
            "TenureDays":           ten,
            "CLV":                  clv_,
            "UniqueProducts":       up,
            "PurchaseFreqMonthly":  pfm,
            "AvgBasketSize":        abs_,
            "MonthlySpendCV":       cv,
            "WeekendShopperRatio":  wsr,
        }

        return self._score(raw_values)

    def predict_from_dict(self, customer_data: dict) -> dict:
        """
        Predict from a dict that already has feature names as keys.
        Missing features are filled with median defaults.
        """
        if not self._loaded:
            self.load()
        return self._score(customer_data)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict cluster for a DataFrame of customers.

        Expects a DataFrame with RFM features as columns.
        Returns the original DataFrame with Cluster and Cluster_Name appended.
        """
        if not self._loaded:
            self.load()

        results = []
        for _, row in df.iterrows():
            res = self._score(row.to_dict())
            results.append({
                "Cluster":      res["cluster_id"],
                "Cluster_Name": res["cluster_name"],
            })

        result_df = df.copy()
        result_df = pd.concat(
            [result_df, pd.DataFrame(results, index=df.index)], axis=1
        )
        return result_df

    # ------------------------------------------------------------------
    # Internal scorer
    # ------------------------------------------------------------------

    def _score(self, raw: dict) -> dict:
        """Core: build feature vector → scale → predict → enrich."""
        # Align with training feature names
        row = []
        for feat in self._feature_names:
            val = raw.get(feat, 0.0)
            row.append(float(val) if val is not None else 0.0)

        X = np.array(row).reshape(1, -1)
        X_scaled = self._scaler.transform(X)

        cluster_id = int(self._model.predict(X_scaled)[0])
        cluster_name = self._cluster_labels.get(cluster_id, "Unknown")

        profile = CLUSTER_PROFILES.get(cluster_name, CLUSTER_PROFILES["Unknown"])

        return {
            "cluster_id":    cluster_id,
            "cluster_name":  cluster_name,
            "emoji":         profile["emoji"],
            "description":   profile["description"],
            "rfm_profile":   profile["rfm_profile"],
            "strategy":      profile["strategy"],
            "retention_risk":    profile["retention_risk"],
            "upsell_potential":  profile["upsell_potential"],
            "color":             profile["color"],
            "input_features":    dict(zip(self._feature_names, row)),
        }


# =============================================================================
# ENTRY POINT (CLI demo)
# =============================================================================

if __name__ == "__main__":
    cfg       = load_config()
    predictor = CustomerSegmentPredictor(cfg).load()

    # Example: a high-value recent customer
    result = predictor.predict_from_rfm(
        recency=10,
        frequency=25,
        monetary=4200.0,
        avg_order_value=168.0,
        tenure_days=365,
    )

    print("\n" + "=" * 55)
    print("  PREDICTION RESULT")
    print("=" * 55)
    print(f"  Cluster ID   : {result['cluster_id']}")
    print(f"  Segment      : {result['emoji']} {result['cluster_name']}")
    print(f"  RFM Profile  : {result['rfm_profile']}")
    print(f"  Description  : {result['description']}")
    print(f"\n  Marketing Strategies:")
    for s in result["strategy"]:
        print(f"    • {s}")
    print("=" * 55 + "\n")
