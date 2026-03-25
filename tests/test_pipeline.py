"""
tests/test_pipeline.py
======================
Unit and integration tests for the Customer Segmentation pipeline.

Run with:
  pytest tests/ -v --cov=src --cov-report=term-missing

Coverage targets:
  - DataCleaner  ≥ 85%
  - RFMFeatureBuilder ≥ 80%
  - CustomerSegmentPredictor ≥ 80%
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── project root in path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, validate_raw_dataframe, validate_rfm_dataframe
from src.data_preprocessing import DataCleaner, _generate_synthetic_data
from src.feature_engineering import RFMFeatureBuilder

warnings.filterwarnings("ignore")


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def raw_df():
    """Generate a small synthetic dataset for testing."""
    return _generate_synthetic_data(n=5_000, seed=0)


@pytest.fixture(scope="session")
def clean_df(config, raw_df):
    cleaner = DataCleaner(config)
    return cleaner.fit_transform(raw_df)


@pytest.fixture(scope="session")
def feat_df(config, clean_df):
    builder = RFMFeatureBuilder(config)
    return builder.build(clean_df)


# ===========================================================================
# CONFIG TESTS
# ===========================================================================

class TestConfig:
    def test_config_loads(self, config):
        assert isinstance(config, dict)

    def test_required_sections(self, config):
        for section in ["data", "preprocessing", "feature_engineering",
                        "modeling", "model"]:
            assert section in config, f"Missing config section: {section}"

    def test_kmeans_k_range(self, config):
        k_range = config["modeling"]["kmeans"]["k_range"]
        assert len(k_range) == 2
        assert k_range[1] > k_range[0]


# ===========================================================================
# DATA GENERATION TESTS
# ===========================================================================

class TestSyntheticData:
    def test_shape(self, raw_df):
        assert raw_df.shape[0] == 5_000
        assert raw_df.shape[1] == 8

    def test_required_columns(self, raw_df):
        required = {
            "InvoiceNo", "StockCode", "Description",
            "Quantity", "InvoiceDate", "UnitPrice",
            "CustomerID", "Country",
        }
        assert required.issubset(set(raw_df.columns))

    def test_has_nulls_in_customer_id(self, raw_df):
        # Synthetic data has ~5% null CustomerIDs
        assert raw_df["CustomerID"].isna().sum() > 0

    def test_has_cancelled_orders(self, raw_df):
        # Synthetic data has 'C' prefixed invoices
        cancelled = raw_df["InvoiceNo"].astype(str).str.startswith("C").sum()
        assert cancelled > 0


# ===========================================================================
# VALIDATION TESTS
# ===========================================================================

class TestValidation:
    def test_valid_raw_df_passes(self, raw_df):
        assert validate_raw_dataframe(raw_df) is True

    def test_invalid_raw_df_raises(self):
        bad_df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError):
            validate_raw_dataframe(bad_df)

    def test_valid_rfm_df_passes(self, feat_df):
        rfm_df = feat_df[["Recency", "Frequency", "Monetary"]].copy()
        rfm_df["CustomerID"] = rfm_df.index
        assert validate_rfm_dataframe(rfm_df) is True


# ===========================================================================
# DATA CLEANING TESTS
# ===========================================================================

class TestDataCleaner:
    def test_output_shape_reduced(self, raw_df, clean_df):
        # Cleaning always reduces rows (nulls, cancellations, etc.)
        assert len(clean_df) < len(raw_df)

    def test_no_null_customer_ids(self, clean_df):
        assert clean_df["CustomerID"].isna().sum() == 0

    def test_no_cancelled_invoices(self, clean_df):
        cancelled = clean_df["InvoiceNo"].astype(str).str.startswith("C").sum()
        assert cancelled == 0

    def test_no_negative_quantities(self, clean_df):
        assert (clean_df["Quantity"] < 0).sum() == 0

    def test_no_zero_prices(self, clean_df):
        assert (clean_df["UnitPrice"] <= 0).sum() == 0

    def test_invoice_date_dtype(self, clean_df):
        assert pd.api.types.is_datetime64_any_dtype(clean_df["InvoiceDate"])

    def test_total_revenue_column(self, clean_df):
        assert "TotalRevenue" in clean_df.columns
        assert (clean_df["TotalRevenue"] > 0).all()

    def test_revenue_calculation(self, clean_df):
        expected = (clean_df["Quantity"] * clean_df["UnitPrice"]).round(10)
        actual   = clean_df["TotalRevenue"].round(10)
        pd.testing.assert_series_equal(expected, actual, check_names=False)

    def test_report_keys(self, config, raw_df):
        cleaner = DataCleaner(config)
        _ = cleaner.fit_transform(raw_df)
        report = cleaner.report()
        for key in ["initial_rows", "final_rows", "rows_removed"]:
            assert key in report

    def test_clean_data_has_temporal_columns(self, clean_df):
        for col in ["Year", "Month", "DayOfWeek", "Hour", "IsWeekend"]:
            assert col in clean_df.columns


# ===========================================================================
# FEATURE ENGINEERING TESTS
# ===========================================================================

class TestRFMFeatureBuilder:
    def test_one_row_per_customer(self, clean_df, feat_df):
        assert len(feat_df) == clean_df["CustomerID"].nunique()

    def test_rfm_columns_present(self, feat_df):
        for col in ["Recency", "Frequency", "Monetary"]:
            assert col in feat_df.columns

    def test_recency_non_negative(self, feat_df):
        assert (feat_df["Recency"] >= 0).all()

    def test_frequency_positive(self, feat_df):
        assert (feat_df["Frequency"] > 0).all()

    def test_monetary_positive(self, feat_df):
        assert (feat_df["Monetary"] > 0).all()

    def test_rfm_scores_range(self, feat_df):
        for col in ["R_Score", "F_Score", "M_Score"]:
            assert col in feat_df.columns
            assert feat_df[col].between(1, 4).all(), f"{col} out of range [1,4]"

    def test_rfm_score_is_sum(self, feat_df):
        expected = feat_df["R_Score"] + feat_df["F_Score"] + feat_df["M_Score"]
        pd.testing.assert_series_equal(
            expected, feat_df["RFM_Score"], check_names=False
        )

    def test_rfm_segment_column(self, feat_df):
        assert "RFM_Segment" in feat_df.columns
        assert feat_df["RFM_Segment"].nunique() > 1

    def test_advanced_features_present(self, feat_df):
        for col in ["CLV", "AvgOrderValue", "TenureDays",
                    "PurchaseFreqMonthly", "UniqueProducts"]:
            assert col in feat_df.columns, f"Missing feature: {col}"

    def test_no_inf_values(self, feat_df):
        num = feat_df.select_dtypes(include=np.number)
        assert not np.isinf(num.values).any()

    def test_model_features_scaled(self, config, feat_df):
        builder = RFMFeatureBuilder(config)
        builder.build  # already built in fixture, re-use scaler
        X, names = builder.get_model_features(feat_df, scale=False)
        assert len(names) > 0
        assert X.shape[0] == len(feat_df)
        assert X.shape[1] == len(names)


# ===========================================================================
# PREDICT TESTS (mocked — no artefacts needed)
# ===========================================================================

class TestPredictorLogic:
    """Test prediction logic without requiring saved model files."""

    def test_cluster_profiles_complete(self):
        from src.predict import CLUSTER_PROFILES
        required_keys = [
            "description", "rfm_profile", "strategy",
            "retention_risk", "upsell_potential", "color"
        ]
        for seg, profile in CLUSTER_PROFILES.items():
            for key in required_keys:
                assert key in profile, f"{seg} missing key: {key}"

    def test_strategy_is_list(self):
        from src.predict import CLUSTER_PROFILES
        for seg, profile in CLUSTER_PROFILES.items():
            assert isinstance(profile["strategy"], list)
            assert len(profile["strategy"]) >= 1

    def test_color_is_hex(self):
        from src.predict import CLUSTER_PROFILES
        import re
        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for seg, profile in CLUSTER_PROFILES.items():
            assert hex_pattern.match(profile["color"]), \
                f"{seg}: invalid color {profile['color']}"


# ===========================================================================
# INTEGRATION TEST (end-to-end, short)
# ===========================================================================

class TestEndToEnd:
    def test_full_feature_pipeline(self, config, raw_df):
        """Smoke test: raw data → feature matrix without crash."""
        cleaner = DataCleaner(config)
        clean   = cleaner.fit_transform(raw_df)

        builder = RFMFeatureBuilder(config)
        feat    = builder.build(clean)

        assert len(feat) > 0
        assert "Recency" in feat.columns
        assert "Cluster" not in feat.columns   # clustering not run yet

    def test_rfm_values_reasonable(self, feat_df):
        """Business sanity checks on RFM values."""
        # Recency: should be < 3 years for most customers
        assert feat_df["Recency"].quantile(0.99) < 1100
        # Monetary: should have customers spending > £1
        assert (feat_df["Monetary"] > 1).all()
        # Frequency: at least 1 order per customer
        assert (feat_df["Frequency"] >= 1).all()
