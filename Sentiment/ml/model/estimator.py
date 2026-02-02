import os
from Sentiment.constant.training_pipeline import SAVED_MODEL_DIR


class ModelResolver:
    """
    Resolves paths for the production inference package.
    Assumes a flat saved_models/ directory (NO versioning).
    """

    MODEL_FILE_NAME = "model.pt"
    TOKENIZER_DIR_NAME = "tokenizer"
    META_FILE_NAME = "meta.yaml"

    def __init__(self, model_dir: str = SAVED_MODEL_DIR):
        self.model_dir = model_dir

    def get_model_path(self) -> str:
        path = os.path.join(self.model_dir, self.MODEL_FILE_NAME)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
        return path

    def get_tokenizer_path(self) -> str:
        path = os.path.join(self.model_dir, self.TOKENIZER_DIR_NAME)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Tokenizer folder not found at {path}")
        return path

    def get_meta_path(self) -> str:
        path = os.path.join(self.model_dir, self.META_FILE_NAME)
        if not os.path.exists(path):
            raise FileNotFoundError(f"meta.yaml not found at {path}")
        return path

    def is_model_exists(self) -> bool:
        return all([
            os.path.exists(os.path.join(self.model_dir, self.MODEL_FILE_NAME)),
            os.path.isdir(os.path.join(self.model_dir, self.TOKENIZER_DIR_NAME)),
            os.path.exists(os.path.join(self.model_dir, self.META_FILE_NAME)),
        ])
