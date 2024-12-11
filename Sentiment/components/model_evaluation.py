from Sentiment.exception import SentimentException
from Sentiment.logger import logging
from Sentiment.utils.main_utils import load_numpy_array_data
from Sentiment.entity.artifact_entity import ModelTrainerArtifact,ModelEvaluationArtifact
from Sentiment.entity.artifact_entity import DataTransformationArtifact
from Sentiment.entity.config_entity import ModelEvaluationConfig
from Sentiment.utils.main_utils import save_object,load_object,write_yaml_file
from Sentiment.ml.model.estimator import ModelResolver
from Sentiment.constant.training_pipeline import TARGET_COLUMN
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import os,sys 
import pandas  as  pd
import numpy as np

class ModelEvaluation:


    def __init__(self,model_eval_config:ModelEvaluationConfig,
                    data_transformation_artifact:DataTransformationArtifact,
                    model_trainer_artifact:ModelTrainerArtifact):
        
        try:
            
            self.model_eval_config=model_eval_config
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
        except Exception as e:
            raise SentimentException(e,sys)
    


    def replace_nan_with_value(self, arr):
        """Replace NaN values in the array with a specified value."""
        arr = np.nan_to_num(arr, nan=0)
        return arr 
    

    def evaluate(self, model, transformed_test_data, transformed_test_attention_masks, test_labels, batch_size=32):
        """Evaluate the model on test data and return loss and accuracy."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # Move test data to the correct device
        test_dataset = TensorDataset(
            torch.tensor(transformed_test_data).to(device),
            torch.tensor(transformed_test_attention_masks).to(device),
            torch.tensor(test_labels).to(device)
        )
    
        val_dataloader = DataLoader(
            test_dataset,
            sampler=RandomSampler(test_dataset),
            batch_size=batch_size
        )
    
        loss_fn = torch.nn.CrossEntropyLoss()
        model.to(device)  # Ensure model is on the correct device
        model.eval()
    
        val_accuracy = []
        val_loss = []
    
        for batch in val_dataloader:
            b_input_ids, b_attn_mask, b_labels = batch  # Already moved to device
    
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)
    
            loss = loss_fn(logits, b_labels)
            val_loss.append(loss.item())
    
            preds = torch.argmax(logits, dim=1).flatten()
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)
    
        avg_loss = np.mean(val_loss)
        avg_accuracy = np.mean(val_accuracy)
        
        return avg_loss, avg_accuracy

    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load the transformed test data and move it to the correct device
            transformed_test_data = torch.tensor(np.load(self.data_transformation_artifact.transformed_test_file_path)).to(device)
            transformed_test_attention_masks = torch.tensor(np.load(self.data_transformation_artifact.transformed_test_attention_mask_path)).to(device)
            test_labels = torch.tensor(np.load(self.data_transformation_artifact.transformed_test_labels_path)).to(device)

            # Get file paths for the trained and latest models
            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()
            is_model_accepted = True

            # Check if there's a previously saved model
            if not model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted, 
                    improved_accuracy=None, 
                    best_model_path=None, 
                    trained_model_path=train_model_file_path, 
                    train_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact, 
                    best_model_metric_artifact=None
                )
                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact

            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(latest_model_path).to(device)  # Ensure model is on the correct device
            train_model = load_object(train_model_file_path).to(device)

            # Evaluate both the new model and the best existing model
            val_loss_train_model, val_accuracy_train_model = self.evaluate(train_model, transformed_test_data, transformed_test_attention_masks, test_labels)
            val_loss_latest_model, val_accuracy_latest_model = self.evaluate(latest_model, transformed_test_data, transformed_test_attention_masks, test_labels)

            # Compare the models' performances
            improved_accuracy = val_accuracy_train_model - val_accuracy_latest_model
            if self.model_eval_config.change_threshold < improved_accuracy:
                is_model_accepted = True
                # Save the new model if it's better
                save_object(latest_model_path, latest_model)
            else:
                is_model_accepted = False

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_path=latest_model_path,
                trained_model_path=train_model_file_path,
                train_model_metric_artifact=val_accuracy_train_model,
                best_model_metric_artifact=val_accuracy_latest_model
            )

            # Log and save evaluation report
            model_eval_report = model_evaluation_artifact.__dict__
            write_yaml_file(self.model_eval_config.report_file_path, model_eval_report)
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise SentimentException(e, sys)
