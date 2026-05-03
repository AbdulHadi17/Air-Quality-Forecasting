"""
LSTM forecasting model for per-station PM2.5 prediction.

Architecture:
    Input → LSTM(128) → Dropout(0.3) → LSTM(64) → Dropout(0.3) →
    Dense(32, ReLU) → Dense(1, Linear)

Supports:
    - Training with early stopping and learning rate reduction
    - Model checkpointing (best weights saved automatically)
    - Training history visualization
    - Prediction and inverse scaling
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from tensorflow.keras.optimizers import Adam

from src.logger import logging
from src.config_loader import get_config


class LSTMForecaster:
    """Stacked LSTM model for per-station time-series forecasting.

    Args:
        config: Project configuration dict. If None, loads defaults.
        input_shape: Tuple of (sequence_length, n_features).
                     If None, inferred from training data.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        input_shape: Optional[tuple[int, int]] = None,
    ):
        self.config = config or get_config()
        self.models_dir = self.config["paths"]["models_dir"]
        self.outputs_dir = self.config["paths"]["outputs_dir"]
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        Path(self.outputs_dir).mkdir(parents=True, exist_ok=True)

        self.model: Optional[Sequential] = None
        self.history = None

        if input_shape is not None:
            self.model = self._build_model(input_shape)

        logging.info("LSTMForecaster initialized")

    def _build_model(self, input_shape: tuple[int, int]) -> Sequential:
        """Build the stacked LSTM architecture.

        Args:
            input_shape: (sequence_length, n_features).

        Returns:
            Compiled Keras Sequential model.
        """
        model = Sequential([
            # Explicit Input layer (required by newer Keras)
            Input(shape=input_shape),

            # First BiLSTM layer — returns sequences for stacking
            Bidirectional(
                LSTM(128, return_sequences=True),
                name="bilstm_1",
            ),
            Dropout(0.3, name="dropout_1"),

            # Second BiLSTM layer — returns final hidden state
            Bidirectional(
                LSTM(64, return_sequences=False),
                name="bilstm_2",
            ),
            Dropout(0.3, name="dropout_2"),

            # Dense head
            Dense(32, activation="relu", name="dense_1"),
            Dropout(0.2, name="dropout_3"),
            Dense(1, activation="linear", name="output"),
        ], name="LSTM_Forecaster")

        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="huber",
            metrics=["mae"],
        )

        # Build before count_params (required in newer Keras versions)
        model.build(input_shape=(None,) + input_shape)
        model.summary(print_fn=lambda x: logging.info(x))
        total_params = model.count_params()
        logging.info(f"LSTM model built: {total_params:,} parameters")

        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 256,
        patience: int = 10,
        validation_split: float = 0.15,
    ) -> dict:
        """Train the LSTM model with callbacks.

        Args:
            X_train: Training sequences (n_samples, seq_length, n_features).
            y_train: Training targets (n_samples,).
            X_val: Optional validation sequences.
            y_val: Optional validation targets.
            epochs: Maximum training epochs.
            batch_size: Batch size.
            patience: Early stopping patience.
            validation_split: Fraction for validation if X_val not provided.

        Returns:
            Training history dictionary.
        """
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self._build_model(input_shape)

        # Callbacks
        checkpoint_path = os.path.join(self.models_dir, "lstm_best.keras")
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
            ModelCheckpoint(
                checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ]

        # Validation data
        validation_data = None
        val_split = 0.0
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            val_split = validation_split

        logging.info(
            f"Training LSTM: X={X_train.shape}, epochs={epochs}, "
            f"batch_size={batch_size}, patience={patience}"
        )

        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=val_split,
            callbacks=callbacks,
            verbose=1,
        )

        best_val_loss = min(self.history.history.get("val_loss", [float("inf")]))
        best_val_mae = min(self.history.history.get("val_mae", [float("inf")]))
        actual_epochs = len(self.history.history["loss"])

        logging.info(
            f"LSTM training complete: {actual_epochs} epochs, "
            f"best val_loss={best_val_loss:.4f}, best val_mae={best_val_mae:.4f}"
        )

        return self.history.history

    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Generate predictions.

        Args:
            X: Input sequences (n_samples, seq_length, n_features).
            batch_size: Prediction batch size.

        Returns:
            Predictions array of shape (n_samples,).
        """
        if self.model is None:
            raise RuntimeError("Model not trained/loaded. Train first or load weights.")

        y_pred = self.model.predict(X, batch_size=batch_size, verbose=0)
        return y_pred.flatten()

    def save(self, path: Optional[str] = None) -> str:
        """Save the full model.

        Args:
            path: Output path. Defaults to saved_models/lstm_final.keras.

        Returns:
            Absolute path to saved model.
        """
        if path is None:
            path = os.path.join(self.models_dir, "lstm_final.keras")

        self.model.save(path)
        logging.info(f"LSTM model saved to {path}")
        return path

    def load(self, path: Optional[str] = None) -> None:
        """Load a saved model.

        Args:
            path: Path to .keras file. Defaults to saved_models/lstm_best.keras.
        """
        if path is None:
            path = os.path.join(self.models_dir, "lstm_best.keras")

        self.model = load_model(path)
        logging.info(f"LSTM model loaded from {path}")
