"""
data_preprocessing.py
=====================
Complete data ingestion, validation, cleaning, and transformation pipeline
for the UCI Online Retail II dataset.

Dataset : UCI Online Retail II
URL     : https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
Records : ~1 million transactions | 8 columns
Period  : Dec 2009 – Dec 2011

Steps
-----
1. Load raw Excel / CSV data
2. Validate schema
3. Handle missing values
4. Remove duplicates & anomalies
5. Filter cancelled orders
6. Outlier treatment (IQR capping)
7. Parse & enrich dates
8. Save cleaned dataset
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (
    PROJECT_ROOT,
    get_logger,
    load_config,
    resolve_path,
    validate_raw_dataframe,
)

warnings.filterwarnings("ignore")

# ── Module-level logger (overridden when run via pipeline) ────────────────────
_log = get_logger(__name__)


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_raw_data(config: dict) -> pd.DataFrame:
    """
    Load the raw Online Retail dataset from disk.

    Supports .xlsx and .csv.  Falls back to generating a synthetic
    dataset if the file is absent (useful for CI / demo purposes).

    Parameters
    ----------
    config : dict – full project config

    Returns
    -------
    pd.DataFrame – raw transactions
    """
    raw_path = PROJECT_ROOT / config["data"]["raw_path"]
    _log.info(f"Loading raw data from: {raw_path}")

    if not raw_path.exists():
        _log.warning(
            f"Raw file not found at {raw_path}. "
            "Generating synthetic demo dataset …"
        )
        return _generate_synthetic_data()

    suffix = raw_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(raw_path, engine="openpyxl")
    elif suffix == ".csv":
        df = pd.read_csv(raw_path, encoding="unicode_escape")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    _log.info(f"Raw data loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def _generate_synthetic_data(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic Online Retail dataset for demo / testing.

    Returns a DataFrame with the same schema as the real dataset.
    """
    rng = np.random.default_rng(seed)
    _log.info(f"Generating {n:,} synthetic transactions …")

    n_customers = 4_000
    n_products = 3_000
    date_range = pd.date_range("2010-12-01", "2011-12-09", freq="h")

    customer_ids = rng.integers(10_000, 18_000, size=n_customers)
    # ~5 % missing customer IDs
    missing_mask = rng.random(n) < 0.05

    quantities = rng.integers(1, 120, size=n)
    # ~2 % cancelled orders (negative quantity)
    cancel_mask = rng.random(n) < 0.02
    quantities[cancel_mask] *= -1

    invoice_nos = [
        (f"C{rng.integers(400_000, 600_000)}" if q < 0
         else str(rng.integers(400_000, 600_000)))
        for q in quantities
    ]

    stock_codes = [f"{rng.integers(10_000, 99_999)}" for _ in range(n)]
    country_list = ["United Kingdom", "Germany", "France", "Spain", "Netherlands",
                    "Belgium", "Switzerland", "Portugal", "Australia", "EIRE"]
    country_raw  = [0.91, 0.02, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005]
    country_p    = [x / sum(country_raw) for x in country_raw]
    countries = rng.choice(country_list, size=n, p=country_p)

    df = pd.DataFrame({
        "InvoiceNo":   invoice_nos,
        "StockCode":   stock_codes,
        "Description": [f"Product {c}" for c in stock_codes],
        "Quantity":    quantities,
        "InvoiceDate": rng.choice(date_range, size=n),
        "UnitPrice":   np.round(rng.uniform(0.50, 50.0, size=n), 2),
        "CustomerID":  [
            (np.nan if miss else float(rng.choice(customer_ids)))
            for miss in missing_mask
        ],
        "Country":     countries,
    })

    # Add a handful of extreme outliers
    outlier_idx = rng.choice(n, size=20, replace=False)
    df.loc[outlier_idx, "Quantity"] = rng.integers(5_000, 80_000, size=20)
    df.loc[outlier_idx[:10], "UnitPrice"] = rng.uniform(500, 2_000, size=10)

    _log.info(f"Synthetic dataset generated: {df.shape}")
    return df


# =============================================================================
# 2. DATA CLEANING PIPELINE
# =============================================================================

class DataCleaner:
    """
    Encapsulates the full cleaning pipeline.

    Usage
    -----
    >>> cleaner = DataCleaner(config)
    >>> clean_df = cleaner.fit_transform(raw_df)
    >>> cleaner.report()
    """

    def __init__(self, config: dict):
        self.config = config
        self.cfg = config.get("preprocessing", {})
        self._report: dict[str, int | str] = {}
        self.logger = get_logger(self.__class__.__name__, config)

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run every cleaning step and return the cleaned DataFrame."""
        self.logger.info("=" * 60)
        self.logger.info("  DATA CLEANING PIPELINE")
        self.logger.info("=" * 60)

        validate_raw_dataframe(df, self.logger)
        df = df.copy()
        self._report["initial_rows"] = len(df)

        df = self._standardise_columns(df)
        df = self._remove_duplicates(df)
        df = self._handle_missing_customer_ids(df)
        df = self._filter_cancelled_orders(df)
        df = self._filter_invalid_quantities(df)
        df = self._filter_invalid_prices(df)
        df = self._remove_test_entries(df)
        df = self._parse_dates(df)
        df = self._treat_outliers(df)
        df = self._add_revenue_column(df)

        self._report["final_rows"] = len(df)
        self._report["rows_removed"] = (
            self._report["initial_rows"] - self._report["final_rows"]
        )
        self._report["unique_customers"] = df["CustomerID"].nunique()
        self._report["date_range"] = (
            f"{df['InvoiceDate'].min().date()} → "
            f"{df['InvoiceDate'].max().date()}"
        )

        self.logger.info(
            f"Cleaning complete: "
            f"{self._report['initial_rows']:,} → "
            f"{self._report['final_rows']:,} rows"
        )
        return df

    # ------------------------------------------------------------------
    # Individual cleaning steps
    # ------------------------------------------------------------------

    def _standardise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace from string columns; normalise column names."""
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(include="string").columns:
            df[col] = df[col].str.strip()
        self.logger.info("Column names and string values standardised.")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop exact duplicate rows."""
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        self._report["duplicates_removed"] = removed
        self.logger.info(f"Duplicates removed: {removed:,}")
        return df

    def _handle_missing_customer_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle rows with null CustomerID.

        Strategy is read from config: 'drop' (default) or future 'impute'.
        """
        before = len(df)
        strategy = self.cfg.get("missing_customer_id", "drop")
        null_count = df["CustomerID"].isna().sum()

        if strategy == "drop":
            df = df.dropna(subset=["CustomerID"])
            self._report["missing_customerid_dropped"] = before - len(df)
            self.logger.info(
                f"Rows with missing CustomerID dropped: {null_count:,}"
            )
        return df

    def _filter_cancelled_orders(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove invoices that represent cancellations (prefix 'C')."""
        if not self.cfg.get("remove_cancelled", True):
            return df
        before = len(df)
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
        removed = before - len(df)
        self._report["cancelled_removed"] = removed
        self.logger.info(f"Cancelled invoices removed: {removed:,}")
        return df

    def _filter_invalid_quantities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where Quantity < minimum threshold."""
        min_qty = self.cfg.get("min_quantity", 1)
        before = len(df)
        df = df[df["Quantity"] >= min_qty]
        self._report["invalid_quantity_removed"] = before - len(df)
        self.logger.info(
            f"Rows with Quantity < {min_qty} removed: {before - len(df):,}"
        )
        return df

    def _filter_invalid_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where UnitPrice < minimum threshold."""
        min_price = self.cfg.get("min_unit_price", 0.01)
        before = len(df)
        df = df[df["UnitPrice"] >= min_price]
        self._report["invalid_price_removed"] = before - len(df)
        self.logger.info(
            f"Rows with UnitPrice < {min_price} removed: {before - len(df):,}"
        )
        return df

    def _remove_test_entries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove known test / non-customer entries.

        Matches StockCodes and Descriptions commonly used for postage,
        bank charges, manual entries, etc.
        """
        if not self.cfg.get("remove_test_customers", True):
            return df

        test_stock_codes = {"POST", "D", "DOT", "M", "BANK CHARGES", "PADS"}
        test_desc_patterns = ["test", "postage", "bank charges", "manual", "AMAZONFEE"]

        before = len(df)

        # Remove known test stock codes
        df = df[~df["StockCode"].str.upper().isin(test_stock_codes)]

        # Remove rows with test descriptions
        desc_lower = df["Description"].str.lower().fillna("")
        for pattern in test_desc_patterns:
            df = df[~desc_lower.str.contains(pattern, na=False)]

        removed = before - len(df)
        self._report["test_entries_removed"] = removed
        self.logger.info(f"Test / non-product entries removed: {removed:,}")
        return df

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse InvoiceDate to datetime and extract temporal features."""
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["Year"]        = df["InvoiceDate"].dt.year
        df["Month"]       = df["InvoiceDate"].dt.month
        df["DayOfWeek"]   = df["InvoiceDate"].dt.day_name()
        df["Hour"]        = df["InvoiceDate"].dt.hour
        df["IsWeekend"]   = df["InvoiceDate"].dt.weekday >= 5

        # Ensure CustomerID is an integer
        df["CustomerID"] = df["CustomerID"].astype(int)

        self.logger.info("Date parsing and temporal feature extraction done.")
        return df

    def _treat_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cap extreme outliers in Quantity and UnitPrice using IQR method.

        Values above Q3 + 3*IQR are capped (Winsorisation) rather than
        dropped to retain as many customers as possible.
        """
        for col in ["Quantity", "UnitPrice"]:
            q1  = df[col].quantile(0.01)
            q3  = df[col].quantile(0.99)
            iqr = q3 - q1
            upper = q3 + 3 * iqr
            n_outliers = (df[col] > upper).sum()
            df[col] = df[col].clip(upper=upper)
            self.logger.info(
                f"Outlier treatment [{col}]: {n_outliers:,} values capped at {upper:.2f}"
            )
        return df

    def _add_revenue_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add TotalRevenue = Quantity × UnitPrice."""
        df["TotalRevenue"] = df["Quantity"] * df["UnitPrice"]
        self.logger.info("TotalRevenue column added.")
        return df

    # ------------------------------------------------------------------
    def report(self) -> dict:
        """Print and return the cleaning summary report."""
        print("\n" + "=" * 55)
        print("  DATA CLEANING REPORT")
        print("=" * 55)
        for k, v in self._report.items():
            print(f"  {k:<38}: {v}")
        print("=" * 55 + "\n")
        return self._report


# =============================================================================
# 3. SAVE CLEANED DATA
# =============================================================================

def save_clean_data(df: pd.DataFrame, config: dict) -> Path:
    """
    Persist the cleaned DataFrame to the processed data directory.

    Saves both CSV (for readability) and Parquet (for performance).
    """
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_path     = processed_dir / "online_retail_clean.csv"
    parquet_path = processed_dir / "online_retail_clean.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    _log.info(f"Clean data saved → {csv_path.name} + {parquet_path.name}")
    return csv_path


# =============================================================================
# 4. ENTRY POINT
# =============================================================================

def run_preprocessing(config: dict | None = None) -> pd.DataFrame:
    """
    Full preprocessing pipeline — load → clean → save.

    Parameters
    ----------
    config : dict, optional  (loads from config.yaml if None)

    Returns
    -------
    pd.DataFrame – cleaned transactions
    """
    if config is None:
        config = load_config()

    raw_df    = load_raw_data(config)
    cleaner   = DataCleaner(config)
    clean_df  = cleaner.fit_transform(raw_df)
    cleaner.report()
    save_clean_data(clean_df, config)
    return clean_df


if __name__ == "__main__":
    cfg = load_config()
    clean = run_preprocessing(cfg)
    print(clean.head())
    print(f"\nShape: {clean.shape}")
    print(f"Customers: {clean['CustomerID'].nunique():,}")
