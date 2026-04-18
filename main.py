"""
Main entry point — runs the full AQI forecasting pipeline end to end.

Usage:
    python main.py                     # Full pipeline (ingestion → train → evaluate)
    python main.py --skip-ingestion    # Skip data download (use existing raw data)
    python main.py --skip-prep         # Skip prep (use existing sequences)
    python main.py --force-refresh     # Force re-download all data
    python main.py --lstm-only         # Skip ConvLSTM (faster iteration)
"""

import argparse
import os
import sys

import numpy as np

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
        "--skip-prep",
        action="store_true",
        help="Skip preprocessing/features/sequences (load existing .npy files)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download all data from APIs",
    )
    parser.add_argument(
        "--lstm-only",
        action="store_true",
        help="Only train LSTM (skip ConvLSTM)",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Only train baseline models",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max training epochs for deep learning models",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for LSTM training",
    )
    args = parser.parse_args()

    logging.info("=" * 60)
    logging.info("  LAHORE AQI FORECASTING — FULL PIPELINE")
    logging.info("=" * 60)

    from src.config_loader import get_config
    config = get_config()

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: DATA PREPARATION
    # ═══════════════════════════════════════════════════════════

    if not args.skip_prep:
        # ── Step 1: Data Ingestion ─────────────────────────
        if not args.skip_ingestion:
            print("\n[1/6] Running data ingestion...")
            from src.data.ingestion_pipeline import DataIngestionPipeline

            pipeline = DataIngestionPipeline()
            ingestion_results = pipeline.run(force_refresh=args.force_refresh)

            if ingestion_results["measurements"].empty:
                print("ERROR: No measurements fetched.")
                sys.exit(1)
        else:
            print("\n[1/6] Skipping data ingestion (using existing data)")

        # ── Step 2: Preprocessing ──────────────────────────
        print("\n[2/6] Running preprocessing...")
        from src.data.preprocess import DataPreprocessor

        preprocessor = DataPreprocessor()
        preprocess_results = preprocessor.run()

        if preprocess_results["merged"].empty:
            print("ERROR: Preprocessing produced empty dataset.")
            sys.exit(1)

        # ── Step 3: Feature Engineering ────────────────────
        print("\n[3/6] Running feature engineering...")
        from src.data.features import FeatureEngineer

        engineer = FeatureEngineer()
        featured_df = engineer.run(
            preprocess_results["merged"],
            preprocess_results["stations_meta"],
        )

        featured_path = os.path.join(config["paths"]["processed_dir"], "featured.parquet")
        featured_df.to_parquet(featured_path, index=False, engine="pyarrow")

        # ── Step 4: Sequence Building ──────────────────────
        print("\n[4/6] Building sequences...")
        from src.data.sequence_builder import SequenceBuilder

        builder = SequenceBuilder()
        lstm_data = builder.build_lstm_sequences(featured_df)

        if not args.lstm_only:
            convlstm_data = builder.build_convlstm_sequences(
                featured_df,
                preprocess_results["stations_meta"],
            )
    else:
        print("\n[1-4/6] Skipping data prep (loading existing sequences)...")
        from src.data.sequence_builder import SequenceBuilder

        seq_dir = config["paths"]["sequences_dir"]
        lstm_data = SequenceBuilder.load_sequences(seq_dir, "lstm")
        if not args.lstm_only:
            convlstm_data = SequenceBuilder.load_sequences(seq_dir, "convlstm")

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: MODEL TRAINING & EVALUATION
    # ═══════════════════════════════════════════════════════════

    print("\n[5/6] Training models...")
    from src.models.metrics import ModelEvaluator
    from src.models.baseline import BaselineModels

    evaluator = ModelEvaluator(config)

    X_train = lstm_data["X_train"]
    y_train = lstm_data["y_train"]
    X_test = lstm_data["X_test"]
    y_test = lstm_data["y_test"]

    print(f"  Training data: X={X_train.shape}, y={y_train.shape}")
    print(f"  Test data:     X={X_test.shape}, y={y_test.shape}")

    # ── Baselines ──────────────────────────────────────────
    print("\n  Training baselines...")
    baselines = BaselineModels(config)
    baseline_preds = baselines.run_all(X_train, y_train, X_test, y_test, evaluator)

    if args.baselines_only:
        evaluator.print_comparison()
        print("\n  Done (baselines only).")
        return

    # ── LSTM ───────────────────────────────────────────────
    print("\n  Training LSTM...")
    from src.models.lstm_model import LSTMForecaster

    lstm = LSTMForecaster(config)
    lstm.train(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    lstm_pred = lstm.predict(X_test)
    evaluator.evaluate(y_test, lstm_pred, "BiLSTM")
    lstm.save()

    # ── ConvLSTM ───────────────────────────────────────────
    if not args.lstm_only:
        print("\n  Training ConvLSTM...")
        from src.models.convlstm_model import ConvLSTMForecaster

        if "X_train" in convlstm_data:
            convlstm = ConvLSTMForecaster(config)
            convlstm.train(
                convlstm_data["X_train"],
                convlstm_data["y_train"],
                epochs=min(args.epochs, 50),
                batch_size=8,
            )
            convlstm_pred = convlstm.predict(convlstm_data["X_test"])
            evaluator.evaluate(
                convlstm_data["y_test"], convlstm_pred, "ConvLSTM"
            )
            convlstm.save()
        else:
            print("  WARNING: ConvLSTM sequences not available, skipping")
    else:
        print("  Skipping ConvLSTM (--lstm-only flag)")

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: COMPARISON
    # ═══════════════════════════════════════════════════════════

    print("\n[6/6] Model comparison...")
    evaluator.print_comparison()

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
