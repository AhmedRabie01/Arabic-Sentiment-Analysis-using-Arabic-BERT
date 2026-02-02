import sys
import os

from Sentiment.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelPusherConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
)

from Sentiment.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
    ModelTrainerArtifact,
)

from Sentiment.exception import SentimentException
from Sentiment.logger import logging

from Sentiment.components.data_ingestion import DataIngestion
from Sentiment.components.data_validation import DataValidation
from Sentiment.components.data_transformation import DataTransformation
from Sentiment.components.model_trainer import ModelTrainer
from Sentiment.components.model_evaluation import ModelEvaluation
from Sentiment.components.model_pusher import ModelPusher

from Sentiment.cloud_strorage.s3_syncer import S3Sync
from Sentiment.constant.s3_bucket import TRAINING_BUCKET_NAME
from Sentiment.constant.training_pipeline import SAVED_MODEL_DIR


class TrainPipeline:
    """
    Orchestrates the complete training lifecycle:
    Ingestion → Validation → Transformation → Training → Evaluation → Pusher
    """

    is_pipeline_running = False

    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()

    # ------------------------------------------------------------------
    # DATA INGESTION
    # ------------------------------------------------------------------
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logging.info("Starting data ingestion")

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            artifact = data_ingestion.initiate_data_ingestion()

            logging.info(f"Data ingestion completed: {artifact}")
            return artifact

        except Exception as e:
            raise SentimentException(e, sys)

    # ------------------------------------------------------------------
    # DATA VALIDATION
    # ------------------------------------------------------------------
    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config,
            )

            artifact = data_validation.initiate_data_validation()
            return artifact

        except Exception as e:
            raise SentimentException(e, sys)

    # ------------------------------------------------------------------
    # DATA TRANSFORMATION
    # ------------------------------------------------------------------
    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config,
            )

            artifact = data_transformation.initiate_data_transformation()
            return artifact

        except Exception as e:
            raise SentimentException(e, sys)

    # ------------------------------------------------------------------
    # MODEL TRAINING
    # ------------------------------------------------------------------
    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_trainer = ModelTrainer(
                model_trainer_config=model_trainer_config,
                data_transformation_artifact=data_transformation_artifact,
            )

            artifact = model_trainer.initiate_model_trainer()
            return artifact

        except Exception as e:
            raise SentimentException(e, sys)

    # ------------------------------------------------------------------
    # MODEL EVALUATION
    # ------------------------------------------------------------------
    def start_model_evaluation(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        try:
            model_eval_config = ModelEvaluationConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_eval = ModelEvaluation(
                model_eval_config=model_eval_config,
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )

            artifact = model_eval.initiate_model_evaluation()
            return artifact

        except Exception as e:
            raise SentimentException(e, sys)

    # ------------------------------------------------------------------
    # MODEL PUSHER (PRODUCTION PROMOTION)
    # ------------------------------------------------------------------
    def start_model_pusher(
        self,
        model_eval_artifact: ModelEvaluationArtifact,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> ModelPusherArtifact:
        try:
            model_pusher_config = ModelPusherConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_pusher = ModelPusher(
                model_pusher_config=model_pusher_config,
                model_eval_artifact=model_eval_artifact,
                data_transformation_artifact=data_transformation_artifact,
            )

            artifact = model_pusher.initiate_model_pusher()
            return artifact

        except Exception as e:
            raise SentimentException(e, sys)

    # ------------------------------------------------------------------
    # OPTIONAL SYNC METHODS
    # ------------------------------------------------------------------
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = (
                f"s3://{TRAINING_BUCKET_NAME}/artifact/"
                f"{self.training_pipeline_config.timestamp}"
            )
            self.s3_sync.sync_folder_to_s3(
                folder=self.training_pipeline_config.artifact_dir,
                aws_buket_url=aws_bucket_url,
            )
        except Exception as e:
            raise SentimentException(e, sys)

    def sync_saved_model_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}"
            self.s3_sync.sync_folder_to_s3(
                folder=SAVED_MODEL_DIR,
                aws_buket_url=aws_bucket_url,
            )
        except Exception as e:
            raise SentimentException(e, sys)

    # ------------------------------------------------------------------
    # RUN PIPELINE
    # ------------------------------------------------------------------
    def run_pipeline(self):
        try:
            if TrainPipeline.is_pipeline_running:
                raise Exception("Training pipeline is already running")

            TrainPipeline.is_pipeline_running = True

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact
            )
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact
            )
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact
            )
            model_eval_artifact = self.start_model_evaluation(
                data_transformation_artifact,
                model_trainer_artifact,
            )

            if not model_eval_artifact.is_model_accepted:
                raise Exception("Trained model is not better than the existing model")

            self.start_model_pusher(
                model_eval_artifact,
                data_transformation_artifact,
            )

            TrainPipeline.is_pipeline_running = False

        except Exception as e:
            TrainPipeline.is_pipeline_running = False
            # Optional: sync artifacts on failure
            # self.sync_artifact_dir_to_s3()
            raise SentimentException(e, sys)
