from Sentiment.ml.model.loader import load_model
import torch


def predict_single_text(text: str):
    model, tokenizer, meta, device = load_model()

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=meta["max_len"],
    ).to(device)

    with torch.no_grad():
        s_logits, i_logits, t_logits = model(**enc)

    s_prob = torch.softmax(s_logits, dim=1)[0]
    i_prob = torch.softmax(i_logits, dim=1)[0]
    t_prob = torch.softmax(t_logits, dim=1)[0]

    s_id = int(s_prob.argmax())
    i_id = int(i_prob.argmax())
    t_id = int(t_prob.argmax())

    return {
        "text": text,
        "sentiment": meta["tasks"]["sentiment"]["labels"][s_id],
        "intent": meta["tasks"]["intent"]["labels"][i_id],
        "topic": meta["tasks"]["topic"]["labels"][t_id],
        "confidence": {
            "sentiment": float(s_prob[s_id] * 100),
            "intent": float(i_prob[i_id] * 100),
            "topic": float(t_prob[t_id] * 100),
        },
    }
