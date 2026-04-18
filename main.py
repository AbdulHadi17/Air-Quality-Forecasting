"""
Main entry point — runs the full AQI forecasting pipeline end to end.

Usage:
    python main.py                     # Full pipeline (ingestion → preprocessing → features → sequences)
    python main.py --skip-ingestion    # Skip data download (use existing raw data)
    python main.py --force-refresh     # Force re-download all data
"""

import argparse
import sys

from src.logger import logging


def main():
    parser = argparse.ArgumentParser(
        description="Lahore AQI Forecasting — Full Pipeline"
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip data ingestion (use existing raw Parquet files)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download all data from APIs",
    )
    parser.add_argument(
        "--lstm-only",
        action="store_true",
        help="Build only LSTM sequences (skip ConvLSTM grid)",
    )
    args = parser.parse_args()

    logging.info("=" * 60)
    logging.info("  LAHORE AQI FORECASTING — FULL PIPELINE")
    logging.info("=" * 60)

    # ── Step 1: Data Ingestion ───────────────────────────────
    if not args.skip_ingestion:
        print("\n[1/4] Running data ingestion...")
        from src.data.ingestion_pipeline import DataIngestionPipeline

        pipeline = DataIngestionPipeline()
        ingestion_results = pipeline.run(force_refresh=args.force_refresh)

        if ingestion_results["measurements"].empty:
            print("ERROR: No measurements fetched. Check API key and connectivity.")
            sys.exit(1)
    else:
        print("\n[1/4] Skipping data ingestion (using existing data)")

    # ── Step 2: Preprocessing ────────────────────────────────
    print("\n[2/4] Running preprocessing...")
    from src.data.preprocess import DataPreprocessor

    preprocessor = DataPreprocessor()
    preprocess_results = preprocessor.run()

    if preprocess_results["merged"].empty:
        print("ERROR: Preprocessing produced empty dataset.")
        sys.exit(1)

    # ── Step 3: Feature Engineering ──────────────────────────
    print("\n[3/4] Running feature engineering...")
    from src.data.features import FeatureEngineer

    engineer = FeatureEngineer()
    featured_df = engineer.run(
        preprocess_results["merged"],
        preprocess_results["stations_meta"],
    )

    # Save featured DataFrame
    import os
    from src.config_loader import get_config
    config = get_config()
    featured_path = os.path.join(config["paths"]["processed_dir"], "featured.parquet")
    featured_df.to_parquet(featured_path, index=False, engine="pyarrow")
    print(f"  Saved featured data: {featured_path}")

    # ── Step 4: Sequence Building ────────────────────────────
    print("\n[4/4] Building sequences...")
    from src.data.sequence_builder import SequenceBuilder

    builder = SequenceBuilder()

    # LSTM sequences
    lstm_result = builder.build_lstm_sequences(featured_df)
    if lstm_result:
        print(f"  LSTM: X_train={lstm_result['X_train'].shape}, "
              f"X_test={lstm_result['X_test'].shape}")

    # ConvLSTM sequences
    if not args.lstm_only:
        convlstm_result = builder.build_convlstm_sequences(
            featured_df,
            preprocess_results["stations_meta"],
        )
        if convlstm_result:
            print(f"  ConvLSTM: X_train={convlstm_result['X_train'].shape}, "
                  f"X_test={convlstm_result['X_test'].shape}")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
