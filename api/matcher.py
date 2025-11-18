# matcher.py

import numpy as np
from .embed_utils import embed_text
from .textextract_utils import split_to_sentences
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Any


def cos_sim_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Core cosine similarity helper for one vector 'a' vs matrix 'b'.
    Returns a 1D array of similarities length = number of rows in b.
    """
    a = a.reshape(1, -1)                    # (1, d)
    sims = cosine_similarity(a, b).reshape(-1)  # (n,)
    return sims


def compute_similarities(jd_text: str, resumes: str | List[str], batch_size: int = 64) -> np.ndarray:
    """
    Shared worker:
      - embeds [jd_text] + resumes in one batch
      - returns a 1-D numpy array of cosine similarities (len = len(resumes))
    """
    if isinstance(resumes, str): # if resumes are str type, then resumes wrap the str in a list else it moves on
        resumes = [resumes]

    if not resumes:
        return np.array([])

    texts = [jd_text] + resumes
    embeddings = embed_text(texts, batch_size=batch_size)  # shape (1 + n, d)

    jd_emb = embeddings[0]       # shape (d,)
    resume_embs = embeddings[1:] # shape (n, d)

    sims = cos_sim_vectors(jd_emb, resume_embs)  # shape (n,)
    return sims

def cosine_similarity_single_resume(jd_text: str, resume_text: str, batch_size: int = 64) -> float:

    sims = compute_similarities(jd_text, resume_text, batch_size=batch_size)
    return float(sims[0]) if sims else 0.0

def rank_resumes(jd_text: str, resume_texts: List[str], top_k: int = None, batch_size: int = 64) -> List[Tuple[int, float]]:
    '''
    for multiple resumes
    '''
    sims = compute_similarities(jd_text, resume_texts, batch_size=batch_size)
    if sims.size == 0:
        return []

    order = np.argsort(-sims)  # descending
    ranked = [(int(i), float(sims[i])) for i in order]
    return ranked[:top_k] if top_k else ranked

# def explain_match_by_sentence(jd_text: str, resume_text: str, top_n: int = 2) -> List[Dict[str, Any]]:
#     """
#     Sentence-level explanation (unchanged): returns top_n resume sentences that match the JD.
#     """
#     jd_emb = embed_text([jd_text])[0]            # shape (d,)
#     sentences = split_to_sentences(resume_text)  # list[str]
#     if not sentences:
#         return []
#     sent_embs = embed_text(sentences)            # shape (m, d)
#     sims = cos_sim_vectors(jd_emb, sent_embs)  # shape (m,)
#     idx = np.argsort(-sims)[:top_n]
#     return [{"sentence": sentences[i]} for i in idx]