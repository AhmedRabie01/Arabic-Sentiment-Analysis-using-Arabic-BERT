import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from Sentiment.exception import SentimentException
from Sentiment.logger import logging
from Sentiment.entity.artifact_entity import ModelTrainerArtifact
from Sentiment.entity.config_entity import ModelTrainerConfig
from Sentiment.ml.model.multitask_bert import MultiTaskBert


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact):
        self.config = model_trainer_config
        self.artifact = data_transformation_artifact
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = "jhu-clsp/mmBERT-base"
        self.n_sentiment = 3
        self.n_intent = 3
        self.n_topic = 7

    def _load(self, path):
        return torch.tensor(np.load(path), dtype=torch.long)

    def _dataloaders(self):
        X_ids = self._load(self.artifact.X_train_ids_path)
        X_mask = self._load(self.artifact.X_train_mask_path)
        y_s = self._load(self.artifact.y_train_sentiment_path)
        y_i = self._load(self.artifact.y_train_intent_path)
        y_t = self._load(self.artifact.y_train_topic_path)

        split = int(0.8 * len(X_ids))

        train_ds = TensorDataset(
            X_ids[:split], X_mask[:split], y_s[:split], y_i[:split], y_t[:split]
        )
        val_ds = TensorDataset(
            X_ids[split:], X_mask[split:], y_s[split:], y_i[split:], y_t[split:]
        )

        return (
            DataLoader(train_ds, RandomSampler(train_ds), batch_size=4),
            DataLoader(val_ds, SequentialSampler(val_ds), batch_size=4),
        )

    def initiate_model_trainer(self):
        try:
            logging.info("Training MultiTaskBERT (state_dict mode)")

            train_dl, _ = self._dataloaders()

            model = MultiTaskBert(
                self.model_name,
                self.n_sentiment,
                self.n_intent,
                self.n_topic,
            ).to(self.device)

            optimizer = AdamW(model.parameters(), lr=2e-5)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 0, len(train_dl)
            )

            loss_fn = nn.CrossEntropyLoss()
            model.train()

            for batch in train_dl:
                ids, mask, ys, yi, yt = [b.to(self.device) for b in batch]
                out = model(ids, mask)

                loss = (
                    loss_fn(out["sentiment"], ys)
                    + loss_fn(out["intent"], yi)
                    + loss_fn(out["topic"], yt)
                )

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)

            # 🔥 THE IMPORTANT LINE 🔥
            torch.save(
                model.state_dict(),
                self.config.trained_model_file_path.replace(".pkl", ".pt")
            )

            return ModelTrainerArtifact(
                trained_model_file_path=self.config.trained_model_file_path.replace(".pkl", ".pt"),
                train_metric_artifact={},
                test_metric_artifact={},
                train_predictions=[],
                val_predictions=[],
            )

        except Exception as e:
            raise SentimentException(e, sys)
