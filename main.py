import sys
import pickle
import os
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, Response
from starlette.middleware.cors import CORSMiddleware
import torch
from Sentiment.pipeline.training_pipeline import TrainPipeline
from Sentiment.constant.training_pipeline import SAVED_MODEL_DIR, MAX_LEN
from Sentiment.exception import SentimentException
from Sentiment.logger import logging
from Sentiment.utils.main_utils import read_yaml_file, load_object
from Sentiment.ml.model.estimator import ModelResolver
from Sentiment.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, \
    DataTransformationConfig
from Sentiment.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from Sentiment.constant.application import APP_HOST, APP_PORT
from Sentiment.components.data_ingestion import DataIngestion
from Sentiment.components.data_validation import DataValidation
from Sentiment.components.data_transformation import DataTransformation

from uvicorn import run as app_run
from transformers import pipeline, AutoModelForSequenceClassification

env_file_path = os.path.join(os.getcwd(), "env.yaml")

def set_env_variable(env_file_path):
    if os.getenv('MONGO_DB_URL', None) is None:
        env_config = read_yaml_file(env_file_path)
        os.environ['MONGO_DB_URL'] = env_config['MONGO_DB_URL']

app = FastAPI()
templates = Jinja2Templates(directory="templates")
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainPipeline()
        if train_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")
        train_pipeline.run_pipeline()
        return Response("Training successful!")
    except Exception as e:
        return Response(f"Error occurred! {e}")
    
@app.post("/predict")
async def predict_route(request: Request):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        form = await request.form()
        x = form["text"]
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return Response("Model is not available")
        
        best_model_path = model_resolver.get_best_model_path()
        with open(best_model_path, "rb") as f:
            bert = pickle.load(f)
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        logging.info("Starting data ingestion")
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")

        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                         data_validation_config=data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()

        data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                 data_transformation_config=data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        tokenizer = load_object(data_transformation_artifact.tokenizer_file_path)
        class_names = ['negative', 'positive']

        encoded_review = tokenizer.encode_plus(
            x,
            max_length=MAX_LEN,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoded_review['input_ids'].to(device)
        attention_mask = encoded_review['attention_mask'].to(device)

        output = bert(input_ids, attention_mask)
        probabilities = torch.softmax(output, dim=1)
        _, prediction = torch.max(output, dim=1)

        sentiment = class_names[prediction]
        percentage = round(100 * torch.max(probabilities).item(), 2)

        print(f'Review text: {x}')
        print(f'Sentiment  : {class_names[prediction]}')

        # Construct the response content with the correct sentiment
        response_content = {
            "text": x,
            "sentiment": sentiment,
            "percentage": percentage
        }

        logging.info(f"Prediction Result: {class_names[prediction]}")  # Add this line for logging

        return JSONResponse(content=response_content)

    except Exception as e:
        logging.error(f"Prediction Error: {e}")  # Add this line for logging
        return Response(f"Error occurred! {e}")

def main():
    try:
        set_env_variable(env_file_path)
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()

    except Exception as e:
        print(f"Error occurred! {e}")

if __name__ == "__main__":
    ##main()
    app_run(app, host=APP_HOST, port=APP_PORT) 