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
from Sentiment.utils.main_utils import save_numpy_array_data
from Sentiment.constant.training_pipeline import SCHEMA_FILE_PATH, META_FILE_PATH, MODEL_NAME, MAX_LEN
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
            self.meta = read_yaml_file(META_FILE_PATH)

            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
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
            max_length=MAX_LEN,
            return_attention_mask=True,
            return_tensors="np",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    @staticmethod
    def _invert_label_map(label_map: dict) -> dict:
        return {v: int(k) for k, v in label_map.items()}

    def _encode_labels_from_meta(self, series: pd.Series, task: str):
        labels = self.meta["tasks"][task]["labels"]
        inv_map = self._invert_label_map(labels)
        encoded = series.map(inv_map).values
        if np.isnan(encoded).any():
            raise Exception(f"Unknown label values found for task '{task}'")
        return encoded.astype(int), inv_map

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
            y_train_sentiment, sentiment_map = self._encode_labels_from_meta(train_df["sentiment"], "sentiment")
            y_train_intent, intent_map = self._encode_labels_from_meta(train_df["intent"], "intent")
            y_train_topic, topic_map = self._encode_labels_from_meta(train_df["topic"], "topic")

            y_test_sentiment = test_df["sentiment"].map(sentiment_map).values
            y_test_intent = test_df["intent"].map(intent_map).values
            y_test_topic = test_df["topic"].map(topic_map).values

            # -------------------------
            # Save artifacts
            # -------------------------
            os.makedirs(self.config.transformed_data_dir, exist_ok=True)

            os.makedirs(self.config.tokenizer_file_path, exist_ok=True)
            self.tokenizer.save_pretrained(self.config.tokenizer_file_path)

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
