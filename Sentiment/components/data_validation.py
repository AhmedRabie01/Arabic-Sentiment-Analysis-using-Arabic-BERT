import os
import sys
import pandas as pd
from typing import Dict

from Sentiment.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from Sentiment.entity.config_entity import DataValidationConfig
from Sentiment.exception import SentimentException
from Sentiment.logger import logging
from Sentiment.utils.main_utils import read_yaml_file, write_yaml_file
from Sentiment.constant.training_pipeline import SCHEMA_FILE_PATH


class DataValidation:
    """
    NLP-aware Data Validation for Multitask Text Classification.
    Validates:
    - Required columns
    - Missing values
    - Minimum samples per label (topic)
    """

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SentimentException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SentimentException(e, sys)

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        required_cols = set(self.schema["required_columns"])
        df_cols = set(df.columns)

        missing_cols = required_cols - df_cols
        if missing_cols:
            raise Exception(f"Missing required columns: {missing_cols}")

    def _validate_nulls(self, df: pd.DataFrame) -> None:
        label_columns = self.schema["label_columns"].keys()

        for col in label_columns:
            if df[col].isnull().any():
                raise Exception(f"Null values found in label column: {col}")

        if df[self.schema["text_column"]].isnull().any():
            raise Exception("Null values found in text column")

    def _validate_text_length(self, df: pd.DataFrame) -> None:
        min_len = self.schema.get("constraints", {}).get("text", {}).get("min_length", 1)

        short_texts = df[self.schema["text_column"]].astype(str).str.len() < min_len
        if short_texts.any():
            count = short_texts.sum()
            raise Exception(f"{count} samples have text shorter than {min_len} characters")

    def _validate_label_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Validate class counts and generate a report.
        """
        report = {}
        min_samples_cfg = self.schema.get("minimum_samples_per_label", {})

        for label, min_count in min_samples_cfg.items():
            value_counts = df[label].value_counts().to_dict()

            report[label] = {
                "class_distribution": value_counts,
                "min_required": min_count,
                "status": "PASS",
            }

            for class_name, count in value_counts.items():
                if count < min_count:
                    report[label]["status"] = "FAIL"
                    report[label]["error"] = (
                        f"Class '{class_name}' has {count} samples "
                        f"(min required: {min_count})"
                    )
                    raise Exception(report[label]["error"])

        return report

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting Data Validation")

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # -------------------------
            # Schema-based validations
            # -------------------------
            self._validate_required_columns(train_df)
            self._validate_required_columns(test_df)

            self._validate_nulls(train_df)
            self._validate_nulls(test_df)

            self._validate_text_length(train_df)
            self._validate_text_length(test_df)

            # -------------------------
            # Label distribution check
            # -------------------------
            drift_report = {
                "train": self._validate_label_distribution(train_df),
                "test": self._validate_label_distribution(test_df),
            }

            # Save validation report
            os.makedirs(
                os.path.dirname(self.data_validation_config.drift_report_file_path),
                exist_ok=True,
            )

            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=drift_report,
            )

            logging.info("Data validation passed successfully")

            return DataValidationArtifact(
                validation_status=True,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

        except Exception as e:
            logging.error(f"Data validation failed: {e}")

            return DataValidationArtifact(
                validation_status=False,
                valid_train_file_path=None,
                valid_test_file_path=None,
                invalid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                invalid_test_file_path=self.data_ingestion_artifact.test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
