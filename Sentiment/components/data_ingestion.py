import os
import sys
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from Sentiment.data_access.twitter_data import TwitterData
from Sentiment.entity.config_entity import DataIngestionConfig
from Sentiment.entity.artifact_entity import DataIngestionArtifact
from Sentiment.exception import SentimentException
from Sentiment.logger import logging
from Sentiment.utils.main_utils import read_yaml_file
from Sentiment.constant.training_pipeline import SCHEMA_FILE_PATH


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.config = data_ingestion_config
            self.schema = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SentimentException(e, sys)

    def _flatten_labels(self, df: DataFrame) -> DataFrame:
        """
        Flatten MongoDB nested label structure:
        labels.sentiment -> sentiment
        labels.intent -> intent
        labels.topic -> topic
        """
        if "labels" in df.columns:
            labels_df = pd.json_normalize(df["labels"])
            df = pd.concat([df.drop(columns=["labels"]), labels_df], axis=1)
        return df

    def export_data_into_feature_store(self) -> DataFrame:
        try:
            logging.info("Exporting data from MongoDB")

            mongo_data = TwitterData()
            df = mongo_data.export_collection_as_dataframe(
                collection_name=self.config.collection_name
            )

            df = self._flatten_labels(df)

            # Enforce schema columns
            required_columns = self.schema["columns"]
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise Exception(f"Missing required columns: {missing_cols}")

            df = df[required_columns]

            # Save feature store
            os.makedirs(os.path.dirname(self.config.feature_store_file_path), exist_ok=True)
            df.to_csv(self.config.feature_store_file_path, index=False)

            logging.info("Feature store created successfully")
            return df

        except Exception as e:
            raise SentimentException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame):
        try:
            train_df, test_df = train_test_split(
                dataframe,
                test_size=self.config.train_test_split_ratio,
                stratify=dataframe["topic"],
                random_state=42,
            )

            os.makedirs(os.path.dirname(self.config.training_file_path), exist_ok=True)

            train_df.to_csv(self.config.training_file_path, index=False)
            test_df.to_csv(self.config.testing_file_path, index=False)

            logging.info("Stratified train/test split completed")

        except Exception as e:
            raise SentimentException(e, sys)
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            df = self.export_data_into_feature_store()
            self.split_data_as_train_test(df)

            return DataIngestionArtifact(
                trained_file_path=self.config.training_file_path,
                test_file_path=self.config.testing_file_path,
            )

        except Exception as e:
            raise SentimentException(e, sys)
