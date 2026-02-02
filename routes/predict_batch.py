from fastapi import APIRouter, UploadFile, File, Query
import torch
import pandas as pd
from routes.schemas import BatchRequest, BatchResponse, PredictionItem, ConfidenceScores
from Sentiment.ml.model.loader import load_model

router = APIRouter(prefix="/predict", tags=["Prediction"])


def _predict_texts(texts):
    model, tokenizer, meta, device = load_model()

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=meta["max_len"],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        s_logits, i_logits, t_logits = model(**enc)

    s_probs = torch.softmax(s_logits, dim=1)
    i_probs = torch.softmax(i_logits, dim=1)
    t_probs = torch.softmax(t_logits, dim=1)

    results = []
    for idx, text in enumerate(texts):
        s_id = int(s_probs[idx].argmax())
        i_id = int(i_probs[idx].argmax())
        t_id = int(t_probs[idx].argmax())

        results.append(
            PredictionItem(
                text=text,
                sentiment=meta["tasks"]["sentiment"]["labels"][s_id],
                intent=meta["tasks"]["intent"]["labels"][i_id],
                topic=meta["tasks"]["topic"]["labels"][t_id],
                confidence=ConfidenceScores(
                    sentiment=float(s_probs[idx][s_id] * 100),
                    intent=float(i_probs[idx][i_id] * 100),
                    topic=float(t_probs[idx][t_id] * 100),
                ),
            )
        )

    return results


@router.post("/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    results = _predict_texts(req.texts)
    return BatchResponse(total=len(results), results=results)


@router.post("/batch/csv", response_model=BatchResponse)
def predict_batch_csv(
    file: UploadFile = File(...),
    text_column: str = Query("text"),
):
    df = pd.read_csv(file.file)
    texts = df[text_column].astype(str).tolist()
    results = _predict_texts(texts)
    return BatchResponse(total=len(results), results=results)
