from dataclasses import dataclass
from typing import Any, Dict, Optional, List


@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    tokenizer_file_path: str

    X_train_ids_path: str
    X_train_mask_path: str

    X_test_ids_path: str
    X_test_mask_path: str

    y_train_sentiment_path: str
    y_train_intent_path: str
    y_train_topic_path: str

    y_test_sentiment_path: str
    y_test_intent_path: str
    y_test_topic_path: str


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: Dict[str, Any]
    test_metric_artifact: Dict[str, Any]
    train_predictions: List
    val_predictions: List


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_accuracy: Optional[float]
    best_model_path: Optional[str]
    trained_model_path: str
    train_model_metric_artifact: Any
    best_model_metric_artifact: Any


@dataclass
class ModelPusherArtifact:
    saved_model_dir: str
    model_file_path: str
    tokenizer_file_path: str
    meta_file_path: str
