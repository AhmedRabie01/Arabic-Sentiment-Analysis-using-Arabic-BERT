import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer,GPT2TokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import sys
from Sentiment.constant.training_pipeline import TARGET_COLUMN, MODEL_NAME, MAX_LEN, BATCH_SIZE
from Sentiment.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from Sentiment.entity.config_entity import DataTransformationConfig
from Sentiment.exception import SentimentException
from Sentiment.logger import logging
from Sentiment.utils.main_utils import save_numpy_array_data, save_object
import torch
import concurrent.futures

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        """
        :param data_validation_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self.label_encoder = LabelEncoder()
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) 
        except Exception as e:
            raise SentimentException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SentimentException(e, sys)

    def tokenize_texts(self, texts):
        # Tokenize texts using the tokenizer
        batch_encoding = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        input_ids = batch_encoding.input_ids
        attention_masks = batch_encoding.attention_mask
        return input_ids, attention_masks
    
    def preprocess_dataset(self, texts, labels, batch_size=16):
        # Tokenize texts in batches using parallel processing
        input_ids = []
        attention_masks = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Define a function to tokenize a single batch of texts
            def tokenize_batch(batch_texts):
                return self.tokenize_texts(batch_texts)

            # Split texts into batches
            batch_texts = np.array_split(texts, len(texts) // batch_size)

            # Tokenize texts in parallel using multiple threads
            results = list(executor.map(tokenize_batch, batch_texts))

        # Collect the tokenized results
        for result in results:
            input_ids.extend(result[0])
            attention_masks.extend(result[1])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(self.label_encoder.transform(labels))

        dataset = TensorDataset(input_ids, attention_masks, labels)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        return dataloader
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            train_texts = train_df["tweet"].tolist()
            train_labels = train_df["label"].tolist()
            test_texts = test_df["tweet"].tolist()
            test_labels = test_df["label"].tolist()
            self.label_encoder.fit(train_labels)

            # Tokenize data
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, truncation=True)
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add padding token

            train_encodings = tokenizer.batch_encode_plus(
                train_texts,
                truncation=True,
                padding='longest',
                max_length=MAX_LEN,
                return_tensors='pt'
            )
            test_encodings = tokenizer.batch_encode_plus(
                test_texts,
                truncation=True,
                padding='longest',
                max_length=MAX_LEN,
                return_tensors='pt'
            )

            # Save the preprocessed data
            save_object(self.data_transformation_config.label_encoder_file_path, self.label_encoder)
            save_object(self.data_transformation_config.tokenizer_file_path, tokenizer)

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_encodings['input_ids'])
            save_numpy_array_data(self.data_transformation_config.transformed_train_attention_mask_path, train_encodings['attention_mask'])
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_encodings['input_ids'])
            save_numpy_array_data(self.data_transformation_config.transformed_test_attention_mask_path, test_encodings['attention_mask'])
            save_numpy_array_data(self.data_transformation_config.transformed_train_labels_path, torch.tensor(self.label_encoder.transform(train_labels)))
            save_numpy_array_data(self.data_transformation_config.transformed_test_labels_path, torch.tensor(self.label_encoder.transform(test_labels)))

            data_transformation_artifact = DataTransformationArtifact(
                tokenizer_file_path=self.data_transformation_config.tokenizer_file_path,
                label_encoder_file_path=self.data_transformation_config.label_encoder_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_train_attention_mask_path=self.data_transformation_config.transformed_train_attention_mask_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_test_attention_mask_path=self.data_transformation_config.transformed_test_attention_mask_path,
                transformed_train_labels_path=self.data_transformation_config.transformed_train_labels_path,
                transformed_test_labels_path=self.data_transformation_config.transformed_test_labels_path
            )

            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise SentimentException(e, sys) from e
