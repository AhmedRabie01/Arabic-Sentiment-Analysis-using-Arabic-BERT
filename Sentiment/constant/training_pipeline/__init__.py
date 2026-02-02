import os

# =====================================================
# GLOBAL PIPELINE SETTINGS
# =====================================================
PIPELINE_NAME: str = "Sentiment"
ARTIFACT_DIR: str = "artifact"
SAVED_MODEL_DIR = os.path.join("saved_models")
# =====================================================
# DATA FILE NAMES
# =====================================================
FILE_NAME: str = "Sentiment.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
# =====================================================
# SCHEMA
# =====================================================
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

# =====================================================
# DATA INGESTION
# =====================================================
DATA_INGESTION_COLLECTION_NAME: str = "total_Reviews"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

# =====================================================
# DATA VALIDATION
# =====================================================
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

# =====================================================
# DATA TRANSFORMATION
# =====================================================
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
BATCH_SIZE: int = 16

# =====================================================
# MODEL TRAINER
# =====================================================
MODEL_NAME: str = "jhu-clsp/mmBERT-base"
MAX_LEN: int = 512

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"

# =====================================================
# MODEL EVALUATION
# =====================================================
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_EVALUATION_REPORT_NAME: str = "report.yaml"

# =====================================================
# MODEL PUSHER
# =====================================================
MODEL_PUSHER_DIR_NAME: str = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR = SAVED_MODEL_DIR
