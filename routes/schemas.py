from pydantic import BaseModel
from typing import List


class ConfidenceScores(BaseModel):
    sentiment: float
    intent: float
    topic: float


class PredictionItem(BaseModel):
    text: str
    sentiment: str
    intent: str
    topic: str
    confidence: ConfidenceScores


class BatchRequest(BaseModel):
    texts: List[str]


class BatchResponse(BaseModel):
    total: int
    results: List[PredictionItem]
