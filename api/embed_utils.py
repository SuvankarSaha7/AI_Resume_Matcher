
#embed_utils.py

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

model_name = "all-MiniLM-L6-v2"
_model = None

def get_model(model_name: str = model_name):
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model

def embed_text(text: List[str], batch_size: int =64, convert_to_numpy: bool = True) -> np.ndarray:
    model=get_model()
    return model.encode(text, batch_size=batch_size, convert_to_numpy=convert_to_numpy, show_progress_bar=False)