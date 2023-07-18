from dataclasses import dataclass

print("Importing RegressionMetricArtifact from artifact_entity.py")

@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path:str
    

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
    label_encoder_file_path: str
    transformed_train_file_path: str
    transformed_train_attention_mask_path: str  # New field for attention masks
    transformed_test_file_path: str
    transformed_test_attention_mask_path: str  # New field for attention masks
    transformed_train_labels_path: str
    transformed_test_labels_path: str
    
@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float

@dataclass
class RegressionMetricArtifact:
    rmse: float


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: float
    test_metric_artifact: float
    train_predictions: list  # Add train predictions field
    val_predictions: list  # Add validation predictions field


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_accuracy: float
    best_model_path: str
    trained_model_path: str
    train_model_metric_artifact: ClassificationMetricArtifact
    best_model_metric_artifact: ClassificationMetricArtifact

@dataclass
class ModelPusherArtifact:
    saved_model_path:str
    model_file_path:str
