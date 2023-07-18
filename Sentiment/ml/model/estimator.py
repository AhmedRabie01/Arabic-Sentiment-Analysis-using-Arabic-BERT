from Sentiment.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME
import os
import torch


class ModelResolver:

    def __init__(self,model_dir=SAVED_MODEL_DIR):
        try:
            self.model_dir = model_dir

        except Exception as e:
            raise e

    def get_best_model_path(self,)->str:
        try:
            timestamps = list(map(int,os.listdir(self.model_dir)))
            latest_timestamp = max(timestamps)
            latest_model_path= os.path.join(self.model_dir,f"{latest_timestamp}",MODEL_FILE_NAME)
            return latest_model_path
        except Exception as e:
            raise e
        
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

    def is_model_exists(self)->bool:
        try:
            if not os.path.exists(self.model_dir):
                return False

            timestamps = os.listdir(self.model_dir)
            if len(timestamps)==0:
                return False
            
            latest_model_path = self.get_best_model_path()

            if not os.path.exists(latest_model_path):
                return False

            return True
        except Exception as e:
            raise e