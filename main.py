from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from routes.predict_batch import router as predict_router
from routes.predict_single import predict_single_text

app = FastAPI(title="Arabic Multitask Sentiment System")

templates = Jinja2Templates(directory="templates")

app.include_router(predict_router)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.post("/predict")
def predict(text: str = Form(...)):
    return JSONResponse(predict_single_text(text))


@app.post("/train")
def train():
    return {"message": "Training should be run from pipeline / Colab"}
