# textextract.py
import os
import re
from typing import List, Optional
from docx import Document
from docx.opc.exceptions import PackageNotFoundError
import zipfile
# ----------------------------
# Fallback regex sentence split
# ----------------------------
def _simple_sentence_split(text: str) -> List[str]:
    if not text:
        return []
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Split on . ? ! followed by a space and capital/number (heuristic)
    parts = re.split(r"(?<=[.?!])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


# ----------------------------
# Lazy import helpers
# ----------------------------
def _import_fitz():
    try:
        import fitz  # PyMuPDF
        return fitz
    except Exception:
        return None


def _import_docx():
    try:
        from docx import Document  # python-docx
        return Document
    except Exception:
        return None


# ----------------------------
# Text extraction functions
# ----------------------------
def extract_text_from_pdf(path: str) -> str:
    fitz = _import_fitz()
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not installed or failed to import.")
    doc = fitz.open(path)
    texts = []
    for p in doc:
        # get_text() may return '' for some pages; join safely
        texts.append(p.get_text() or "")
    return "\n".join(texts)

    
def _is_valid_docx(path: str) -> bool:
    """Quick binary check: real .docx files start with PK\x03\x04 (ZIP header)."""
    try:
        with open(path, "rb") as f:
            sig = f.read(4)
        return sig == b"PK\x03\x04"
    except Exception:
        return False

def extract_text_from_docx(path: str, allow_fallback_txt: bool = False) -> str:
    """
    Safely extract text from a .docx file.

    - If file doesn't look like a .docx ZIP, raises RuntimeError.
    - Tries python-docx; converts its errors to RuntimeError with a clear message.
    - If allow_fallback_txt=True, will attempt to read raw text as a fallback.
    """
    # quick binary signature check
    if not _is_valid_docx(path):
        raise RuntimeError(f"Not a valid .docx package: {path}")

    Document = _import_docx()
    if Document is None:
        raise RuntimeError("python-docx is not installed or failed to import.")

    try:
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except PackageNotFoundError as e:
        raise RuntimeError(f"Package not found when opening .docx: {e}")
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Corrupt .docx (bad zip): {e}")
    except Exception as e:
        # optional fallback to plain-text read if caller wants it
        if allow_fallback_txt:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception:
                raise RuntimeError(f"Failed to open .docx and fallback failed: {e}")
        raise RuntimeError(f"Unexpected error extracting .docx: {e}")


def extract_text_from_txt(path: str, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        return f.read()


def extract_text_auto(path: str) -> str:
    """
    Detect file extension and extract text accordingly.
    Supports .pdf, .docx, .txt. Falls back to reading as text for unknown extensions.
    """
    p = (path or "").lower()
    if p.endswith(".pdf"):
        return extract_text_from_pdf(path)
    elif p.endswith(".docx"):
        return extract_text_from_docx(path)
    elif p.endswith(".txt"):
        return extract_text_from_txt(path)
    # fallback: try reading as text
    return extract_text_from_txt(path)


# ----------------------------
# Robust sentence splitter
# ----------------------------
def split_to_sentences(text: str) -> List[str]:
    """
    Safe sentence splitter:
      - Tries NLTK sent_tokenize
      - If punkt is missing -> attempts to download it into venv or user dir
      - If NLTK or download fails -> fallback to regex splitter
    Returns list[str] where each sentence is stripped and only short empty tokens are removed.
    """
    if not text:
        return []

    # Lazy import NLTK to avoid import-time errors
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
    except Exception:
        # NLTK not installed -> fallback
        return _simple_sentence_split(text)

    # Try tokenization normally
    try:
        sents = sent_tokenize(text)
        return [s.strip() for s in sents if len(s.strip()) > 3]

    except LookupError:
        # punkt resource missing -> try to download it into venv or user dir
        try:
            venv_dir = os.environ.get("VIRTUAL_ENV")
            if venv_dir:
                nltk_data_dir = os.path.join(venv_dir, "nltk_data")
                os.makedirs(nltk_data_dir, exist_ok=True)
                nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)
                if nltk_data_dir not in nltk.data.path:
                    nltk.data.path.append(nltk_data_dir)
            else:
                nltk.download("punkt", quiet=True)

            # retry tokenization
            sents = sent_tokenize(text)
            return [s.strip() for s in sents if len(s.strip()) > 3]

        except Exception:
            # If download fails -> fallback
            return _simple_sentence_split(text)

    except Exception:
        # Any other tokenization error -> fallback
        return _simple_sentence_split(text)


# ----------------------------
# Optional small helper: safe_extract_and_split
# ----------------------------
def safe_extract_and_split(path: str) -> List[str]:
    """
    Convenience helper: extract text from the given path and split to sentences.
    Returns list[str] (may be empty). Converts None to empty string safely.
    """
    try:
        text = extract_text_auto(path) or ""
    except Exception:
        # On extraction failure, return empty list instead of raising (useful in batch jobs)
        return []
    return split_to_sentences(text)

