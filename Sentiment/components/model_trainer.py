import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from Sentiment.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from Sentiment.entity.config_entity import ModelTrainerConfig
from Sentiment.exception import SentimentException
from Sentiment.utils.main_utils import save_object, load_object
from Sentiment.constant.training_pipeline import MODEL_NAME
from Sentiment.logger import logging
import numpy as np

class BertClassifier(nn.Module):
    def __init__(self, tokenizer=None, freeze_bert=False):
        super(BertClassifier, self).__init__()
        self.tokenizer = tokenizer
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        D_in = self.bert.config.hidden_size

        # Classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(D_in, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),  # Final output layer
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            raise SentimentException(e, sys)

    def criterion(self):
        return nn.CrossEntropyLoss()

    def initialize_model(self, epochs, tokenizer):
        try:
            roberta_classifier = BertClassifier(tokenizer=tokenizer)
            roberta_classifier.to(self.device)

            optimizer = AdamW(roberta_classifier.parameters(), lr=1e-5, no_deprecation_warning=True)
            total_steps = epochs * len(self.data_transformation_artifact.transformed_train_file_path)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

            return roberta_classifier, optimizer, scheduler
        except Exception as e:
            raise SentimentException("Error initializing the model", sys) from e

    def predict(self, model, dataloader):
        model.eval()
        predictions = []

        for batch in dataloader:
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)

            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                predictions.extend(torch.argmax(probs, dim=1).cpu().tolist())

        return predictions

    def train_model(self, train_data, train_attention_masks, train_labels, num_epochs=10, batch_size=8, validation_split=0.2):
        train_size = int(len(train_data) * (1 - validation_split))
        val_size = len(train_data) - train_size
    
        train_dataset = TensorDataset(
            train_data[:train_size], train_attention_masks[:train_size], train_labels[:train_size]
        )
        val_dataset = TensorDataset(
            train_data[train_size:], train_attention_masks[train_size:], train_labels[train_size:]
        )
    
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    
        tokenizer = load_object(self.data_transformation_artifact.tokenizer_file_path)
        model, optimizer, scheduler = self.initialize_model(num_epochs, tokenizer)
        loss_fn = self.criterion()
    
        for epoch in range(num_epochs):
            model.train()
            total_loss, train_preds, train_actuals = 0, [], []
    
            for batch in train_dataloader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                optimizer.zero_grad()
    
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
    
                total_loss += loss.item()
                train_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                train_actuals.extend(labels.cpu().tolist())
    
            train_accuracy = accuracy_score(train_actuals, train_preds)
            train_f1 = f1_score(train_actuals, train_preds, average='weighted')
    
            # Validation phase
            model.eval()
            val_preds, val_actuals = [], []
    
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                    logits = model(input_ids, attention_mask)
                    val_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                    val_actuals.extend(labels.cpu().tolist())
    
            val_accuracy = accuracy_score(val_actuals, val_preds)
            val_f1 = f1_score(val_actuals, val_preds, average='weighted')
    
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss/len(train_dataloader):.4f}, "
                  f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}, "
                  f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
    
        return model, train_preds, val_preds, {'accuracy': train_accuracy, 'f1': train_f1}, {'accuracy': val_accuracy, 'f1': val_f1}


    def initiate_model_trainer(self):
        try:
            train_data = torch.tensor(np.load(self.data_transformation_artifact.transformed_train_file_path)).to(self.device)
            train_attention_masks = torch.tensor(np.load(self.data_transformation_artifact.transformed_train_attention_mask_path)).to(self.device)
            train_labels = torch.tensor(np.load(self.data_transformation_artifact.transformed_train_labels_path)).to(self.device)

            model, train_preds, val_preds, train_metrics, val_metrics = self.train_model(train_data, train_attention_masks, train_labels)

            # Save the trained model
            save_object(self.model_trainer_config.trained_model_file_path, model)

            # Create the artifact
            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=val_metrics,
                train_predictions=train_preds,
                val_predictions=val_preds
            )
        except Exception as e:
            raise SentimentException(e, sys) from e

