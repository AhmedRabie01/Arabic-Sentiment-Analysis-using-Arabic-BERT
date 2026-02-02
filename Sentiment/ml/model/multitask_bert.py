import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class MultiTaskBert(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_sentiment: int,
        n_intent: int,
        n_topic: int,
        dropout: float = 0.2,
        init_from_pretrained: bool = False,  # IMPORTANT
    ):
        super().__init__()

        self.model_name = model_name

        if init_from_pretrained:
            # trust_remote_code lets us load architectures (e.g., ModernBERT/mmBERT)
            # that may not be bundled with the installed transformers version.
            self.encoder = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
        else:
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            self.encoder = AutoModel.from_config(
                config,
                trust_remote_code=True,
            )

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(hidden_size, n_sentiment)
        self.intent_head = nn.Linear(hidden_size, n_intent)
        self.topic_head = nn.Linear(hidden_size, n_topic)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled = outputs.last_hidden_state[:, 0]  # CLS
        pooled = self.dropout(pooled)

        return (
            self.sentiment_head(pooled),
            self.intent_head(pooled),
            self.topic_head(pooled),
        )
