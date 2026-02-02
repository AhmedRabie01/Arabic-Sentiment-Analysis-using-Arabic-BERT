import os
from datetime import datetime
from Sentiment.constant import training_pipeline
from Sentiment.constant.training_pipeline import ARTIFACT_DIR,PIPELINE_NAME


class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name: str = training_pipeline.PIPELINE_NAME
        self.artifact_dir: str = os.path.join(training_pipeline.ARTIFACT_DIR, timestamp)
        self.timestamp: str = timestamp




class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME,
        )

        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FILE_NAME,
        )

        self.training_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME,
        )

        self.testing_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME,
        )

        self.train_test_split_ratio = (
            training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        )

        self.collection_name = training_pipeline.DATA_INGESTION_COLLECTION_NAME
class DataValidationConfig:

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME,
        )

        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_VALID_DIR,
        )

        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_INVALID_DIR,
        )

        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.TRAIN_FILE_NAME,
        )

        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.TEST_FILE_NAME,
        )

        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TRAIN_FILE_NAME,
        )

        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TEST_FILE_NAME,
        )

        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )


class DataTransformationConfig:
    def __init__(self, training_pipeline_config):
        base_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME,
        )

        self.transformed_data_dir = os.path.join(base_dir, "transformed")

        self.tokenizer_file_path = os.path.join(base_dir, "tokenizer")

        self.X_train_ids_path = os.path.join(self.transformed_data_dir, "X_train_ids.npy")
        self.X_train_mask_path = os.path.join(self.transformed_data_dir, "X_train_mask.npy")

        self.X_test_ids_path = os.path.join(self.transformed_data_dir, "X_test_ids.npy")
        self.X_test_mask_path = os.path.join(self.transformed_data_dir, "X_test_mask.npy")

        self.y_train_sentiment_path = os.path.join(self.transformed_data_dir, "y_train_sentiment.npy")
        self.y_train_intent_path = os.path.join(self.transformed_data_dir, "y_train_intent.npy")
        self.y_train_topic_path = os.path.join(self.transformed_data_dir, "y_train_topic.npy")

        self.y_test_sentiment_path = os.path.join(self.transformed_data_dir, "y_test_sentiment.npy")
        self.y_test_intent_path = os.path.join(self.transformed_data_dir, "y_test_intent.npy")
        self.y_test_topic_path = os.path.join(self.transformed_data_dir, "y_test_topic.npy")
# Sentiment/entity/config_entity.py

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config):
        self.model_trainer_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_TRAINER_DIR_NAME
        )

        self.trained_model_file_path = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_NAME
        )


class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config):
        self.model_evaluation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_EVALUATION_DIR_NAME
        )
        self.report_file_path = os.path.join(
            self.model_evaluation_dir,
            training_pipeline.MODEL_EVALUATION_REPORT_NAME
        )
        self.change_threshold = training_pipeline.MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE


class ModelPusherConfig:
    def __init__(self, training_pipeline_config):
        self.saved_model_dir = training_pipeline.MODEL_PUSHER_SAVED_MODEL_DIR
        self.saved_model_file_path = os.path.join(self.saved_model_dir, "model.pt")
        self.saved_meta_file_path = os.path.join(self.saved_model_dir, "meta.yaml")
        self.saved_tokenizer_dir = os.path.join(self.saved_model_dir, "tokenizer")
        self.meta_template_path = training_pipeline.META_FILE_PATH
