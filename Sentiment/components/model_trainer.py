import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from sklearn.metrics import accuracy_score
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from Sentiment.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from Sentiment.entity.config_entity import ModelTrainerConfig
from Sentiment.exception import SentimentException
from Sentiment.utils.main_utils import save_object, load_object
from Sentiment.constant.training_pipeline import MODEL_NAME
import numpy as np
from Sentiment.logger import logging
from numpy import ndarray


class BertClassifier(nn.Module):
    def __init__(self, tokenizer=None, freeze_aragpt=False):
        super(BertClassifier, self).__init__()

        D_in = 768  # Update the input dimension based on the aragpt-base model
        H, D_out = 50, 2  # Update the intermediate layer size and output dimension as needed
        self.tokenizer = tokenizer
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        if freeze_aragpt:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        if input_ids.size(0) == 0:
            raise ValueError("Empty input_ids tensor")
        if attention_mask.size(0) == 0:
            raise ValueError("Empty attention_mask tensor")

        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SentimentException(e)

    def criterion(self):
        # Define the loss function
        return nn.CrossEntropyLoss()

    def initialize_model(self, epochs, tokenizer):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the BertClassifier model
        bert_classifier = BertClassifier(tokenizer=tokenizer)
        # Tell PyTorch to run the model on GPU
        bert_classifier.to(device)

        print("Tokenizer's maximum sequence length:", tokenizer.model_max_length)

        # Create the optimizer
        optimizer = AdamW(params=list(bert_classifier.parameters()),
                          lr=5e-5,    # Default learning rate
                          eps=1e-8    # Default epsilon value
                          )
        total_steps = len(self.data_transformation_artifact.transformed_train_file_path) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        return bert_classifier, optimizer, scheduler

    def predict(self, model, dataloader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        predictions = []

        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)

            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                predicted_labels = torch.argmax(probabilities, dim=1)
                predictions.extend(predicted_labels.cpu().tolist())

        return predictions

    def train_model(self, transformed_train_data: ndarray, transformed_train_attention_masks: ndarray,
                    train_labels: ndarray, num_epochs=1, batch_size=16, validation_split=0.2):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # Split the data into train and validation sets
        train_size = int(len(transformed_train_data) * (1 - validation_split))
        train_dataset = TensorDataset(
            torch.tensor(transformed_train_data[:train_size]),
            torch.tensor(transformed_train_attention_masks[:train_size]),
            torch.tensor(train_labels[:train_size])
        )
        val_dataset = TensorDataset(
            torch.tensor(transformed_train_data[train_size:]),
            torch.tensor(transformed_train_attention_masks[train_size:]),
            torch.tensor(train_labels[train_size:])
        )
    
        # Create train and validation DataLoaders
        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=batch_size,
            drop_last=False  # Set drop_last to False to keep the last incomplete batch
        )
        val_dataloader = DataLoader(
            val_dataset,
            sampler=RandomSampler(val_dataset),
            batch_size=batch_size,
            drop_last=False  # Set drop_last to False to keep the last incomplete batch
        )
        # Load the tokenizer object from the saved file
        tokenizer = load_object(self.data_transformation_artifact.tokenizer_file_path)

        # Check the vocabulary size of the tokenizer
        vocab_size = tokenizer.vocab_size
        print("Vocabulary size:", vocab_size)

        # Initialize the BertClassifier model, optimizer, and scheduler
        bert_classifier, optimizer, scheduler = self.initialize_model(num_epochs, tokenizer)

        # Training loop
        for epoch in range(num_epochs):
            # Set model to training mode
            bert_classifier.train()
            total_loss = 0

            for batch in train_dataloader:
                # Step 1: Retrieve input data and labels from the batch
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)

                # Check the range of values in the input_ids tensor
                max_input_id = input_ids.max().item()
                min_input_id = input_ids.min().item()
                print("Max input ID:", max_input_id)
                print("Min input ID:", min_input_id)

                # Check if tokenizer and model use the same vocabulary
                if vocab_size != tokenizer.vocab_size:
                    raise ValueError("Mismatched vocabularies between tokenizer and model")


                # Step 2: Zero the gradients
                optimizer.zero_grad()

                # Step 3: Forward pass
                logits = bert_classifier(input_ids, attention_mask)

                # Step 4: Compute the loss
                loss = self.criterion()(logits, labels)

                # Step 5: Backward pass
                loss.backward()

                # Step 6: Update model parameters
                optimizer.step()
                scheduler.step()

                # Step 7: Accumulate loss value
                total_loss += loss.item()

                print('finsh 1')

            # Calculate average loss for the epoch
            avg_loss = total_loss / len(train_dataloader)
            print('finsh 1')

            # Compute train accuracy as the train metric artifact
            train_predictions = self.predict(bert_classifier, train_dataloader)
            train_accuracy = accuracy_score(train_labels[:train_size], train_predictions)

            # Compute validation accuracy as the validation metric artifact
            val_predictions = self.predict(bert_classifier, val_dataloader)
            val_accuracy = accuracy_score(train_labels[train_size:], val_predictions)

            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}, Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy}")

        return bert_classifier, train_accuracy, val_accuracy, train_predictions, val_predictions

    def initiate_model_trainer(self) -> DataTransformationArtifact:
        try:
            # Load the transformed train data, attention masks, and labels
            transformed_train_data = np.load(self.data_transformation_artifact.transformed_train_file_path)
            transformed_train_attention_masks = np.load(
                self.data_transformation_artifact.transformed_train_attention_mask_path)
            train_labels = np.load(self.data_transformation_artifact.transformed_train_labels_path)

            model, train_accuracy, val_accuracy, train_predictions, val_predictions = self.train_model(
                transformed_train_data, transformed_train_attention_masks, train_labels
            )

            # Overfitting and Underfitting
            diff = abs(train_accuracy - val_accuracy)

            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model is not good, try to do more experimentation.")

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            save_object(self.model_trainer_config.trained_model_file_path, obj=model)

            # Create and return the model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_accuracy,  # Train accuracy as the train metric artifact
                test_metric_artifact=val_accuracy,  # Validation accuracy as the validation metric artifact
                train_predictions=train_predictions,
                val_predictions=val_predictions
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise SentimentException(e)
