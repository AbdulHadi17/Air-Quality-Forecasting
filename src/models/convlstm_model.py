"""
Hybrid ConvLSTM model for spatio-temporal PM2.5 forecasting.

Captures both spatial correlations (via Conv2D kernels) and temporal
dynamics (via LSTM recurrence) across the Lahore monitoring grid.

Architecture:
    Input (T, H, W, C) → ConvLSTM2D(64) → BN → Dropout →
    ConvLSTM2D(32)      → BN → Dropout →
    Conv2D(16, 3×3)     → ReLU →
    Conv2D(1, 1×1)      → Linear Output (H, W)

Where:
    T = sequence_length (24 hours)
    H, W = spatial grid dimensions
    C = number of features (PM2.5 + weather)
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    ConvLSTM2D,
    BatchNormalization,
    Dropout,
    Conv2D,
    Reshape,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from tensorflow.keras.optimizers import Adam

from src.logger import logging
from src.config_loader import get_config


class ConvLSTMForecaster:
    """Hybrid ConvLSTM model for grid-based spatio-temporal forecasting.

    Args:
        config: Project configuration dict. If None, loads defaults.
        input_shape: Tuple (seq_length, grid_H, grid_W, n_features).
                     If None, inferred from training data.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        input_shape: Optional[tuple[int, int, int, int]] = None,
    ):
        self.config = config or get_config()
        self.models_dir = self.config["paths"]["models_dir"]
        self.outputs_dir = self.config["paths"]["outputs_dir"]
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        Path(self.outputs_dir).mkdir(parents=True, exist_ok=True)

        self.model: Optional[Model] = None
        self.history = None

        if input_shape is not None:
            self.model = self._build_model(input_shape)

        logging.info("ConvLSTMForecaster initialized")

    def _build_model(
        self,
        input_shape: tuple[int, int, int, int],
    ) -> Model:
        """Build the ConvLSTM architecture.

        Args:
            input_shape: (seq_length, grid_H, grid_W, n_features).

        Returns:
            Compiled Keras Model.
        """
        seq_len, grid_h, grid_w, n_features = input_shape

        inputs = Input(shape=input_shape, name="input_grid")

        # ── ConvLSTM Block 1 ─────────────────────────────────
        x = ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
            name="convlstm_1",
        )(inputs)
        x = BatchNormalization(name="bn_1")(x)
        x = Dropout(0.3, name="dropout_1")(x)

        # ── ConvLSTM Block 2 ─────────────────────────────────
        x = ConvLSTM2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=False,  # Output last timestep only
            activation="relu",
            name="convlstm_2",
        )(x)
        x = BatchNormalization(name="bn_2")(x)
        x = Dropout(0.3, name="dropout_2")(x)

        # ── Spatial Conv head ─────────────────────────────────
        # Shape here: (batch, grid_H, grid_W, 32)
        x = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            name="conv_1",
        )(x)

        # Output: 1 channel per grid cell = predicted PM2.5
        outputs = Conv2D(
            filters=1,
            kernel_size=(1, 1),
            padding="same",
            activation="linear",
            name="output_conv",
        )(x)

        # Squeeze last dimension: (batch, H, W, 1) → (batch, H, W)
        outputs = Reshape((grid_h, grid_w), name="output")(outputs)

        model = Model(inputs=inputs, outputs=outputs, name="ConvLSTM_Forecaster")

        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="huber",
            metrics=["mae"],
        )

        model.summary(print_fn=lambda x: logging.info(x))
        total_params = model.count_params()
        logging.info(f"ConvLSTM model built: {total_params:,} parameters")

        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 16,
        patience: int = 8,
        validation_split: float = 0.15,
    ) -> dict:
        """Train the ConvLSTM model with callbacks.

        Args:
            X_train: Training grid sequences (n, T, H, W, C).
            y_train: Training target grids (n, H, W).
            X_val: Optional validation sequences.
            y_val: Optional validation targets.
            epochs: Maximum training epochs.
            batch_size: Batch size (small due to large tensors).
            patience: Early stopping patience.
            validation_split: Fraction for validation if X_val not given.

        Returns:
            Training history dictionary.
        """
        if self.model is None:
            input_shape = X_train.shape[1:]  # (T, H, W, C)
            self.model = self._build_model(input_shape)

        # Callbacks
        checkpoint_path = os.path.join(self.models_dir, "convlstm_best.keras")
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
                patience=4,
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
            f"Training ConvLSTM: X={X_train.shape}, epochs={epochs}, "
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
        actual_epochs = len(self.history.history["loss"])

        logging.info(
            f"ConvLSTM training complete: {actual_epochs} epochs, "
            f"best val_loss={best_val_loss:.4f}"
        )

        return self.history.history

    def predict(self, X: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """Generate grid predictions.

        Args:
            X: Input grid sequences (n, T, H, W, C).
            batch_size: Prediction batch size.

        Returns:
            Predicted grids of shape (n, H, W).
        """
        if self.model is None:
            raise RuntimeError("Model not trained/loaded.")

        return self.model.predict(X, batch_size=batch_size, verbose=0)

    def save(self, path: Optional[str] = None) -> str:
        """Save the full model.

        Args:
            path: Output path. Defaults to saved_models/convlstm_final.keras.

        Returns:
            Absolute path to saved model.
        """
        if path is None:
            path = os.path.join(self.models_dir, "convlstm_final.keras")

        self.model.save(path)
        logging.info(f"ConvLSTM model saved to {path}")
        return path

    def load(self, path: Optional[str] = None) -> None:
        """Load a saved model.

        Args:
            path: Path to .keras file. Defaults to saved_models/convlstm_best.keras.
        """
        if path is None:
            path = os.path.join(self.models_dir, "convlstm_best.keras")

        self.model = load_model(path)
        logging.info(f"ConvLSTM model loaded from {path}")
