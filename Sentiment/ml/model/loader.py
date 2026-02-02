import os
import sys
import yaml
import torch
from transformers import AutoTokenizer
from Sentiment.ml.model.multitask_bert import MultiTaskBert

MODEL_DIR = "saved_models"
_CACHE = {"model": None, "tokenizer": None, "meta": None, "device": None}

# ---------------------------------------------------------------------------
# ModernBERT uses torch.compile during import; torch.compile is not supported
# on Windows. Patch it to a no-op early to avoid runtime import failures.
# ---------------------------------------------------------------------------
if sys.platform.startswith("win") and hasattr(torch, "compile"):
    def _noop_compile(fn=None, *args, **kwargs):
        # Handles both @torch.compile and @torch.compile(...)
        if fn is None:
            def decorator(f):
                return f
            return decorator
        return fn
    torch.compile = _noop_compile


def load_model():
    if _CACHE["model"] is not None:
        return _CACHE["model"], _CACHE["tokenizer"], _CACHE["meta"], _CACHE["device"]

    force_cpu = os.getenv("SENTIMENT_FORCE_CPU", "").lower() in {"1", "true", "yes"}
    requested = os.getenv("SENTIMENT_DEVICE", "").lower()
    if force_cpu or requested == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(MODEL_DIR, "model.pt")
    meta_path = os.path.join(MODEL_DIR, "meta.yaml")
    tokenizer_dir = os.path.join(MODEL_DIR, "tokenizer")

    if not os.path.exists(model_path):
        raise RuntimeError("model.pt not found - train the model first")

    if not os.path.exists(meta_path):
        raise RuntimeError("meta.yaml not found")

    if not os.path.isdir(tokenizer_dir):
        raise RuntimeError("tokenizer folder not found")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    meta = _normalize_meta(meta)

    # Build on CPU first to avoid GPU OOM spikes during state_dict load
    model = MultiTaskBert(
        meta["model_name"],
        len(meta["tasks"]["sentiment"]["labels"]),
        len(meta["tasks"]["intent"]["labels"]),
        len(meta["tasks"]["topic"]["labels"]),
        init_from_pretrained=False,   # VERY IMPORTANT
    )

    state_dict = torch.load(model_path, map_location="cpu")

    # Backward-compat: older checkpoints used prefix "bert." and head names without suffix.
    if any(key.startswith("bert.") for key in state_dict.keys()):
        remapped = {}
        for key, val in state_dict.items():
            if key.startswith("bert."):
                remapped["encoder." + key[len("bert."):]] = val
            elif key.startswith("sentiment"):
                remapped["sentiment_head" + key[len("sentiment"):]] = val
            elif key.startswith("intent"):
                remapped["intent_head" + key[len("intent"):]] = val
            elif key.startswith("topic"):
                remapped["topic_head" + key[len("topic"):]] = val
            else:
                remapped[key] = val
        state_dict = remapped

    # Drop any weights whose shape doesn't match the current architecture
    current_state = model.state_dict()
    filtered_state = {}
    dropped = []
    for key, val in state_dict.items():
        if key in current_state and current_state[key].shape != val.shape:
            dropped.append(key)
            continue
        filtered_state[key] = val

    if dropped:
        # If heads were dropped, the model will reinit them randomly.
        print(
            f"[load_model] dropped incompatible keys: {', '.join(dropped[:5])}"
            f"{' ...' if len(dropped) > 5 else ''}"
        )

    model.load_state_dict(filtered_state, strict=False)

    if device.type == "cuda":
        try:
            model.to(device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("[load_model] CUDA OOM; falling back to CPU")
                device = torch.device("cpu")
                model.to(device)
                if hasattr(torch, "cuda"):
                    torch.cuda.empty_cache()
            else:
                raise

    model.eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    except Exception:
        # Fallback to hub tokenizer (local dump is incomplete)
        tokenizer = AutoTokenizer.from_pretrained(
            meta["model_name"],
            trust_remote_code=True,
        )

    _CACHE.update({"model": model, "tokenizer": tokenizer, "meta": meta, "device": device})
    return _CACHE["model"], _CACHE["tokenizer"], _CACHE["meta"], _CACHE["device"]


def _labels_to_mapping(labels):
    if isinstance(labels, dict):
        # Ensure keys are ints when possible
        mapping = {}
        for k, v in labels.items():
            try:
                mapping[int(k)] = v
            except Exception:
                mapping[k] = v
        return mapping
    if isinstance(labels, list):
        return {i: v for i, v in enumerate(labels)}
    raise ValueError("labels must be list or dict")


def _normalize_meta(meta):
    """
    Supports two formats:
    1) tasks: { sentiment: {labels: {0:..}}, ... }, max_len
    2) labels: { sentiment: [..], ... }, max_length
    """
    if meta is None:
        raise ValueError("meta.yaml is empty")

    if "tasks" in meta:
        tasks = meta["tasks"]
        norm_tasks = {}
        for task, cfg in tasks.items():
            labels = cfg.get("labels", cfg)
            norm_tasks[task] = {"labels": _labels_to_mapping(labels)}
        meta["tasks"] = norm_tasks
    elif "labels" in meta:
        norm_tasks = {}
        for task, labels in meta["labels"].items():
            norm_tasks[task] = {"labels": _labels_to_mapping(labels)}
        meta["tasks"] = norm_tasks
    else:
        raise ValueError("meta.yaml must contain 'tasks' or 'labels'")

    if "max_len" not in meta:
        if "max_length" in meta:
            meta["max_len"] = meta["max_length"]
        else:
            meta["max_len"] = 256

    return meta
