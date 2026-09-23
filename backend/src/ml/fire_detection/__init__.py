"""Model-agnostic ML feature representation and training data for Fire Detection.

This package is the ML training foundation for User Story 2.2's active
wildfire detection. It is intentionally independent of
FireDetectionCalculator (the deterministic rule-based baseline) and of
FireDetectionAgent (runtime orchestration): it does not fetch external APIs,
query Neon, persist FireEvents, or control agents/simulation orchestration.
"""

from src.ml.fire_detection.fire_detection_feature_extractor import FireDetectionFeatureExtractor
from src.ml.fire_detection.fire_detection_features import (
    FIRE_DETECTION_FEATURE_NAMES,
    FireDetectionFeatures,
)
from src.ml.fire_detection.fire_detection_training_data_generator import (
    DATASET_VERSION_V1,
    DATASET_VERSION_V2,
    DEFAULT_TRAINING_DATA_SAMPLE_COUNT,
    DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V2,
    DEFAULT_TRAINING_DATA_SEED,
    FireDetectionTrainingDataGenerator,
    FireDetectionTrainingSample,
)

__all__ = [
    "FIRE_DETECTION_FEATURE_NAMES",
    "FireDetectionFeatures",
    "FireDetectionFeatureExtractor",
    "FireDetectionTrainingDataGenerator",
    "FireDetectionTrainingSample",
    "DEFAULT_TRAINING_DATA_SEED",
    "DEFAULT_TRAINING_DATA_SAMPLE_COUNT",
    "DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V2",
    "DATASET_VERSION_V1",
    "DATASET_VERSION_V2",
]
