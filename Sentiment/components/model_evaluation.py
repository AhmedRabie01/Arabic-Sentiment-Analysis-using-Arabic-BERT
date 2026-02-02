import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

from Sentiment.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
    DataTransformationArtifact,
)
from Sentiment.entity.config_entity import ModelEvaluationConfig
from Sentiment.exception import SentimentException
from Sentiment.logger import logging
from Sentiment.utils.main_utils import load_object, write_yaml_file
from Sentiment.ml.model.estimator import ModelResolver


class ModelEvaluation:
    """
    Multitask-aware Model Evaluation.
    Focuses on:
    - Macro F1 per task
    - Class imbalance handling
    - Composite decision score
    """

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.config = model_eval_config
            self.transform_artifact = data_transformation_artifact
            self.trainer_artifact = model_trainer_artifact
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            raise SentimentException(e, sys)

    def _load_test_data(self):
        X_ids = torch.tensor(np.load(self.transform_artifact.X_test_ids_path)).to(self.device)
        X_mask = torch.tensor(np.load(self.transform_artifact.X_test_mask_path)).to(self.device)

        y_sent = torch.tensor(np.load(self.transform_artifact.y_test_sentiment_path)).to(self.device)
        y_int = torch.tensor(np.load(self.transform_artifact.y_test_intent_path)).to(self.device)
        y_top = torch.tensor(np.load(self.transform_artifact.y_test_topic_path)).to(self.device)

        ds = TensorDataset(X_ids, X_mask, y_sent, y_int, y_top)
        return DataLoader(ds, sampler=SequentialSampler(ds), batch_size=32)

    def _evaluate_model(self, model, dataloader):
        model.eval()

        sent_preds, int_preds, top_preds = [], [], []
        sent_true, int_true, top_true = [], [], []

        with torch.no_grad():
            for batch in dataloader:
                ids, mask, ys, yi, yt = batch
                s_logits, i_logits, t_logits = model(ids, mask)

                sent_preds.extend(torch.argmax(s_logits, dim=1).cpu().numpy())
                int_preds.extend(torch.argmax(i_logits, dim=1).cpu().numpy())
                top_preds.extend(torch.argmax(t_logits, dim=1).cpu().numpy())

                sent_true.extend(ys.cpu().numpy())
                int_true.extend(yi.cpu().numpy())
                top_true.extend(yt.cpu().numpy())

        metrics = {
            "sentiment": {
                "accuracy": accuracy_score(sent_true, sent_preds),
                "macro_f1": f1_score(sent_true, sent_preds, average="macro"),
            },
            "intent": {
                "accuracy": accuracy_score(int_true, int_preds),
                "macro_f1": f1_score(int_true, int_preds, average="macro"),
            },
            "topic": {
                "accuracy": accuracy_score(top_true, top_preds),
                "macro_f1": f1_score(top_true, top_preds, average="macro"),
            },
        }

        return metrics

    def _composite_score(self, metrics: dict) -> float:
        return (
            0.3 * metrics["sentiment"]["macro_f1"]
            + 0.3 * metrics["intent"]["macro_f1"]
            + 0.4 * metrics["topic"]["macro_f1"]
        )

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting Model Evaluation")

            dataloader = self._load_test_data()

            new_model = load_object(self.trainer_artifact.trained_model_file_path).to(self.device)
            new_metrics = self._evaluate_model(new_model, dataloader)
            new_score = self._composite_score(new_metrics)

            resolver = ModelResolver()
            is_accepted = True
            best_score = None
            best_model_path = None

            if resolver.is_model_exists():
                best_model_path = resolver.get_best_model_path()
                best_model = load_object(best_model_path).to(self.device)
                best_metrics = self._evaluate_model(best_model, dataloader)
                best_score = self._composite_score(best_metrics)

                if new_score <= best_score + self.config.change_threshold:
                    is_accepted = False

            report = {
                "new_model_metrics": new_metrics,
                "new_composite_score": new_score,
                "best_composite_score": best_score,
                "accepted": is_accepted,
                "decision_rule": "Weighted macro-F1 (topic prioritized)",
            }

            os.makedirs(os.path.dirname(self.config.report_file_path), exist_ok=True)
            write_yaml_file(self.config.report_file_path, report)

            return ModelEvaluationArtifact(
                is_model_accepted=is_accepted,
                improved_accuracy=new_score - best_score if best_score else None,
                best_model_path=best_model_path,
                trained_model_path=self.trainer_artifact.trained_model_file_path,
                train_model_metric_artifact=new_metrics,
                best_model_metric_artifact=None,
            )

        except Exception as e:
            raise SentimentException(e, sys)
