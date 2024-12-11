import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import sys
from Sentiment.constant.training_pipeline import (
    TARGET_COLUMN, MODEL_NAME, MAX_LEN, BATCH_SIZE, FEATURE_COLUMN
)
from Sentiment.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from Sentiment.entity.config_entity import DataTransformationConfig
from Sentiment.exception import SentimentException
from Sentiment.logger import logging
from Sentiment.utils.main_utils import save_numpy_array_data, save_object
import emoji
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.data.path.append('C:\\Users\\ahmed\\AppData\\Roaming\\nltk_data')  # Update if required
nltk.download('punkt')
nltk.download('punkt_tab')



class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self.label_encoder = LabelEncoder()
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.model_max_length = MAX_LEN
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        except Exception as e:
            raise SentimentException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SentimentException(f"Error reading data from {file_path}. Exception: {str(e)}", sys)

    @staticmethod
    def preprocess_text(text):
        try:
            text = text.lower()
            text = re.sub(r'http\S+|www\S+', '', text)
            text = re.sub(r'\@\w+|\#', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            text = emoji.demojize(text)
            return ' '.join(word_tokenize(text))
        except Exception as e:
            raise SentimentException(f"Error in text preprocessing. Text: {text[:30]}... Exception: {str(e)}", sys)

    def tokenize_texts(self, texts):
        try:
            preprocessed_texts = [self.preprocess_text(text) for text in texts]
            batch_encoding = self.tokenizer.batch_encode_plus(
                preprocessed_texts,
                add_special_tokens=True,
                truncation=True,
                padding='max_length',
                max_length=MAX_LEN,
                return_tensors='pt'
            )
            return batch_encoding.input_ids, batch_encoding.attention_mask
        except Exception as e:
            raise SentimentException(f"Error tokenizing texts. Exception: {str(e)}", sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            train_texts, train_labels = train_df[FEATURE_COLUMN].tolist(), train_df[TARGET_COLUMN].tolist()
            test_texts, test_labels = test_df[FEATURE_COLUMN].tolist(), test_df[TARGET_COLUMN].tolist()

            self.label_encoder.fit(train_labels)
            train_input_ids, train_attention_masks = self.tokenize_texts(train_texts)
            test_input_ids, test_attention_masks = self.tokenize_texts(test_texts)

            save_object(self.data_transformation_config.label_encoder_file_path, self.label_encoder)
            save_object(self.data_transformation_config.tokenizer_file_path, self.tokenizer)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_input_ids)
            save_numpy_array_data(self.data_transformation_config.transformed_train_attention_mask_path, train_attention_masks)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_input_ids)
            save_numpy_array_data(self.data_transformation_config.transformed_test_attention_mask_path, test_attention_masks)
            save_numpy_array_data(self.data_transformation_config.transformed_train_labels_path, torch.tensor(self.label_encoder.transform(train_labels)))
            save_numpy_array_data(self.data_transformation_config.transformed_test_labels_path, torch.tensor(self.label_encoder.transform(test_labels)))

            return DataTransformationArtifact(
                tokenizer_file_path=self.data_transformation_config.tokenizer_file_path,
                label_encoder_file_path=self.data_transformation_config.label_encoder_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_train_attention_mask_path=self.data_transformation_config.transformed_train_attention_mask_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_test_attention_mask_path=self.data_transformation_config.transformed_test_attention_mask_path,
                transformed_train_labels_path=self.data_transformation_config.transformed_train_labels_path,
                transformed_test_labels_path=self.data_transformation_config.transformed_test_labels_path
            )
        except Exception as e:
            raise SentimentException(f"Error during data transformation initiation. Exception: {str(e)}", sys)
