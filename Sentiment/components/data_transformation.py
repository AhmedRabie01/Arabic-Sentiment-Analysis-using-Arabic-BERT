import os
import sys
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer

from Sentiment.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from Sentiment.entity.config_entity import DataTransformationConfig
from Sentiment.exception import SentimentException
from Sentiment.logger import logging
from Sentiment.utils.main_utils import save_numpy_array_data, save_object
from Sentiment.constant.training_pipeline import SCHEMA_FILE_PATH
from Sentiment.utils.main_utils import read_yaml_file


class DataTransformation:
    """
    Multitask NLP Data Transformation:
    - Shared tokenizer
    - Shared text encoding
    - Separate label tensors
    """

    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            self.validation_artifact = data_validation_artifact
            self.config = data_transformation_config
            self.schema = read_yaml_file(SCHEMA_FILE_PATH)

            self.tokenizer = AutoTokenizer.from_pretrained(
                "jhu-clsp/mmBERT-base",
                use_fast=True,
            )

        except Exception as e:
            raise SentimentException(e, sys)

    @staticmethod
    def _read_csv(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def _encode_texts(self, texts):
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors="np",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    @staticmethod
    def _encode_labels(series: pd.Series):
        """
        Encode categorical labels to integer indices.
        Mapping is derived from sorted unique values.
        """
        classes = sorted(series.unique())
        mapping = {label: idx for idx, label in enumerate(classes)}
        encoded = series.map(mapping).values
        return encoded, mapping

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting Data Transformation")

            train_df = self._read_csv(self.validation_artifact.valid_train_file_path)
            test_df = self._read_csv(self.validation_artifact.valid_test_file_path)

            text_col = self.schema["text_column"]

            # -------------------------
            # Encode text (shared)
            # -------------------------
            X_train_ids, X_train_mask = self._encode_texts(train_df[text_col].tolist())
            X_test_ids, X_test_mask = self._encode_texts(test_df[text_col].tolist())

            # -------------------------
            # Encode labels (separately)
            # -------------------------
            y_train_sentiment, sentiment_map = self._encode_labels(train_df["sentiment"])
            y_train_intent, intent_map = self._encode_labels(train_df["intent"])
            y_train_topic, topic_map = self._encode_labels(train_df["topic"])

            y_test_sentiment = test_df["sentiment"].map(sentiment_map).values
            y_test_intent = test_df["intent"].map(intent_map).values
            y_test_topic = test_df["topic"].map(topic_map).values

            # -------------------------
            # Save artifacts
            # -------------------------
            os.makedirs(self.config.transformed_data_dir, exist_ok=True)

            save_object(self.config.tokenizer_file_path, self.tokenizer)

            save_numpy_array_data(self.config.X_train_ids_path, X_train_ids)
            save_numpy_array_data(self.config.X_train_mask_path, X_train_mask)
            save_numpy_array_data(self.config.X_test_ids_path, X_test_ids)
            save_numpy_array_data(self.config.X_test_mask_path, X_test_mask)

            save_numpy_array_data(self.config.y_train_sentiment_path, y_train_sentiment)
            save_numpy_array_data(self.config.y_train_intent_path, y_train_intent)
            save_numpy_array_data(self.config.y_train_topic_path, y_train_topic)

            save_numpy_array_data(self.config.y_test_sentiment_path, y_test_sentiment)
            save_numpy_array_data(self.config.y_test_intent_path, y_test_intent)
            save_numpy_array_data(self.config.y_test_topic_path, y_test_topic)

            logging.info("Data Transformation completed successfully")

            return DataTransformationArtifact(
                tokenizer_file_path=self.config.tokenizer_file_path,

                X_train_ids_path=self.config.X_train_ids_path,
                X_train_mask_path=self.config.X_train_mask_path,

                X_test_ids_path=self.config.X_test_ids_path,
                X_test_mask_path=self.config.X_test_mask_path,

                y_train_sentiment_path=self.config.y_train_sentiment_path,
                y_train_intent_path=self.config.y_train_intent_path,
                y_train_topic_path=self.config.y_train_topic_path,

                y_test_sentiment_path=self.config.y_test_sentiment_path,
                y_test_intent_path=self.config.y_test_intent_path,
                y_test_topic_path=self.config.y_test_topic_path,
            )

        except Exception as e:
            raise SentimentException(e, sys)
