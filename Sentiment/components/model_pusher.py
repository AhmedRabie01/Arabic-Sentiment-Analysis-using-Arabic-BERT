import os
import sys
import shutil

from Sentiment.exception import SentimentException
from Sentiment.logger import logging
from Sentiment.entity.artifact_entity import (
    ModelPusherArtifact,
    ModelEvaluationArtifact,
    DataTransformationArtifact,
)
from Sentiment.entity.config_entity import ModelPusherConfig


class ModelPusher:
    """
    FINAL DEPLOYMENT STAGE (CRITICAL)

    Copies ONLY:
      - model.pkl
      - tokenizer.pkl
      - meta.yaml

    Into:
      saved_models/

    Always overwrite.
    No timestamps.
    No versioning.
    """

    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_eval_artifact: ModelEvaluationArtifact,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.config = model_pusher_config
            self.model_eval_artifact = model_eval_artifact
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SentimentException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("Starting Model Pusher (production overwrite mode)")

            # Safety gate
            if not self.model_eval_artifact.is_model_accepted:
                raise Exception("Model not accepted by evaluation. Aborting deployment.")

            trained_model_path = self.model_eval_artifact.trained_model_path
            tokenizer_src = self.data_transformation_artifact.tokenizer_file_path
            meta_src = self.config.meta_template_path

            if not os.path.exists(trained_model_path):
                raise Exception(f"Trained model not found: {trained_model_path}")

            if not os.path.exists(tokenizer_src):
                raise Exception(f"Tokenizer not found: {tokenizer_src}")

            if not os.path.exists(meta_src):
                raise Exception(
                    f"Meta file not found at {meta_src}. "
                    f"Create config/meta.yaml before training."
                )

            # Ensure production dir exists
            os.makedirs(self.config.saved_model_dir, exist_ok=True)

            # Copy model
            shutil.copyfile(trained_model_path, self.config.saved_model_file_path)
            logging.info(f"Model deployed → {self.config.saved_model_file_path}")

            # Copy tokenizer
            shutil.copyfile(tokenizer_src, self.config.saved_tokenizer_file_path)
            logging.info(f"Tokenizer deployed → {self.config.saved_tokenizer_file_path}")

            # Copy meta.yaml
            shutil.copyfile(meta_src, self.config.saved_meta_file_path)
            logging.info(f"Meta deployed → {self.config.saved_meta_file_path}")

            return ModelPusherArtifact(
                saved_model_dir=self.config.saved_model_dir,
                model_file_path=self.config.saved_model_file_path,
                tokenizer_file_path=self.config.saved_tokenizer_file_path,
                meta_file_path=self.config.saved_meta_file_path,
            )

        except Exception as e:
            raise SentimentException(e, sys)
