"""
run_pipeline.py
===============
Master pipeline script — runs the full end-to-end pipeline:

  Step 1: Data preprocessing
  Step 2: Feature engineering
  Step 3: Model training & evaluation
  Step 4: Print business summary

Usage
-----
  python run_pipeline.py                    # full run
  python run_pipeline.py --step preprocess  # single step
  python run_pipeline.py --step features
  python run_pipeline.py --step train
"""

import argparse
import time
from pathlib import Path

from src.utils import load_config, get_logger, pretty_print_dict
from src.data_preprocessing import run_preprocessing
from src.feature_engineering import run_feature_engineering
from src.train_model import run_training

cfg = load_config()
log = get_logger("Pipeline", cfg)

BANNER = """
╔══════════════════════════════════════════════════════════╗
║          CUSTOMER SEGMENTATION ML PIPELINE               ║
║          UCI Online Retail II · scikit-learn             ║
╚══════════════════════════════════════════════════════════╝
"""


def main(step: str = "all") -> None:
    print(BANNER)
    t0 = time.time()

    if step in ("all", "preprocess"):
        log.info("=" * 55)
        log.info("  STEP 1: DATA PREPROCESSING")
        log.info("=" * 55)
        t = time.time()
        clean_df = run_preprocessing(cfg)
        log.info(f"Step 1 done in {time.time()-t:.1f}s | "
                 f"Shape: {clean_df.shape}")

    if step in ("all", "features"):
        if step == "features":
            # Load previously cleaned data
            import pandas as pd
            from src.utils import PROJECT_ROOT
            clean_path = PROJECT_ROOT / "data" / "processed" / "online_retail_clean.parquet"
            if not clean_path.exists():
                log.error("Cleaned data not found. Run --step preprocess first.")
                return
            clean_df = pd.read_parquet(clean_path)

        log.info("=" * 55)
        log.info("  STEP 2: FEATURE ENGINEERING")
        log.info("=" * 55)
        t = time.time()
        feat, X_scaled, feat_names = run_feature_engineering(clean_df, cfg)
        log.info(f"Step 2 done in {time.time()-t:.1f}s | "
                 f"Features: {len(feat_names)} | Customers: {len(feat)}")

    if step in ("all", "train"):
        if step == "train":
            import pandas as pd
            from src.utils import PROJECT_ROOT
            feat = pd.read_csv(
                PROJECT_ROOT / "data" / "processed" / "rfm_features.csv",
                index_col=0,
            )
            X_scaled = pd.read_csv(
                PROJECT_ROOT / "data" / "processed" / "rfm_scaled.csv",
                index_col=0,
            )
            feat_names = list(X_scaled.columns)

        log.info("=" * 55)
        log.info("  STEP 3: CLUSTERING & EVALUATION")
        log.info("=" * 55)
        t = time.time()
        feat_labelled, all_labels = run_training(feat, X_scaled, cfg)
        log.info(f"Step 3 done in {time.time()-t:.1f}s")

        # Business summary
        print("\n" + "=" * 65)
        print("  BUSINESS INSIGHTS — CLUSTER SUMMARY")
        print("=" * 65)
        summary = (
            feat_labelled
            .groupby("Cluster_Name")
            .agg(
                Customers=("Recency", "count"),
                Avg_Recency=("Recency", "mean"),
                Avg_Frequency=("Frequency", "mean"),
                Total_Revenue=("Monetary", "sum"),
                Avg_CLV=("CLV", "mean"),
            )
            .round(1)
            .sort_values("Total_Revenue", ascending=False)
        )
        print(summary.to_string())

    total_time = time.time() - t0
    log.info(f"\n✅ Pipeline complete in {total_time:.1f}s")
    print(f"\n✅ All plots saved to docs/eda_plots/")
    print(f"✅ Models saved to models/")
    print(f"✅ Processed data saved to data/processed/")
    print(f"\nStart the Streamlit app:")
    print(f"  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the segmentation pipeline")
    parser.add_argument(
        "--step",
        choices=["all", "preprocess", "features", "train"],
        default="all",
        help="Which pipeline step to run (default: all)",
    )
    args = parser.parse_args()
    main(args.step)
