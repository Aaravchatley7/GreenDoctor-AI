"""Configuration module for Explainable AI Plant Disease Detection.

Defines all paths, dataset split ratios, model hyperparameters, early stopping settings,
and execution options using strong type annotations.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    """Central configuration class for the training and evaluation pipeline."""

    # Project Directories
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    DATASET_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "DATASET")
    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    MODELS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "models")
    LOGS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "logs")
    REPORTS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "reports")

    # Data Split Ratios (70% Train, 15% Val, 15% Test)
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    SEED: int = 42

    # Image & Preprocessing Specs
    IMAGE_SIZE: tuple[int, int] = (224, 224)
    IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Model & Transfer Learning Settings
    MODEL_NAME: str = "efficientnet_b0"
    PRETRAINED: bool = True
    STAGE1_EPOCHS: int = 5
    STAGE2_EPOCHS: int = 15
    MAX_TOTAL_EPOCHS: int = 20

    # Hyperparameters
    BATCH_SIZE: int = 32
    STAGE1_LR: float = 1e-4
    STAGE2_LR: float = 1e-5
    WEIGHT_DECAY: float = 1e-4
    NUM_WORKERS: int = 0  # Recommended for macOS

    # Early Stopping & Scheduler
    EARLY_STOPPING_PATIENCE: int = 5
    SCHEDULER_PATIENCE: int = 2
    SCHEDULER_FACTOR: float = 0.5

    # Acceleration
    USE_AMP: bool = False  # Set to False on Apple Silicon MPS

    # Checkpoint & Artifact Filenames inside MODELS_DIR
    BEST_MODEL_NAME: str = "best_model.pth"
    LAST_MODEL_NAME: str = "last_model.pth"
    CLASS_NAMES_FILE: str = "class_names.json"
    HISTORY_FILE: str = "training_history.json"
    OPTIMIZER_STATE_FILE: str = "optimizer_state.pth"

    def create_directories(self) -> None:
        """Creates all required output directories if they do not exist."""
        for path in [self.DATA_DIR, self.MODELS_DIR, self.LOGS_DIR, self.REPORTS_DIR]:
            path.mkdir(parents=True, exist_ok=True)
