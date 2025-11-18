# api.py
import tempfile
import re
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .textextract_utils import extract_text_auto, split_to_sentences
from .matcher import rank_resumes, cosine_similarity_single_resume
from typing import List
import os

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]

FRONTEND_ORIGIN=os.getenv('FRONTEND_ORIGIN', 'http://localhost:5173')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],  # use ["*"] ONLY for temporary debug
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

MAX_FILE_SIZE = 50 * 1024 * 1024  # 15 MB (increase if you really need to, but stay < 25 MB for Vercel Pro)

async def save_upload_to_tempfile(upload: UploadFile) -> str:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    if not upload.filename.lower().endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf, .docx, .txt files are allowed")

    suffix = os.path.splitext(upload.filename)[1]  # keeps the correct extension

    path = tempfile.mktemp(suffix=suffix)

    size = 0
    CHUNK_SIZE = 1024 * 1024  # 1 MiB chunks

    try:
        with open(path, "wb") as f:
            async for chunk in upload:          # ← this is the correct async streaming way
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,  # Payload Too Large
                        detail=f"File '{upload.filename}' too large (max {MAX_FILE_SIZE // (1024*1024)} MB)"
                    )
                f.write(chunk)
        if size == 0:
            os.unlink(path)
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        return path
    except Exception:
        if os.path.exists(path):
            os.unlink(path)
        raise


def extract_email(text: str):
    m = EMAIL_RE.search(text)
    return m.group(0) if m else None

# Replace your current extract_name_simple with this function.

def extract_name_simple(text: str, original_filename: str = None):
    """
    Robust name extraction:
      1. Try spaCy NER (PERSON) if available and returns multi-token person.
      2. Try top-line heuristics (ALL-CAPS, title-case, initials, hyphenated).
      3. Try email-derived name with smart splitting of mashed local-parts.
      4. Try nameparser (if installed) to normalize the chosen candidate.
      5. Fallback to filename or 'Unknown'.
    """
    # helpers ----------------------------------------------------------
    def try_spacy(text_snip: str):
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text_snip[:12000])  # cap length
            persons = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
            if not persons:
                return None
            # prefer the longest meaningful PERSON entity
            best = max(persons, key=lambda s: (len(s.split()), len(s)))
            return normalize_name(best)
        except Exception:
            return None

    def normalize_name(raw: str):
        # remove extraneous punctuation and multiple spaces, preserve hyphens and apostrophes
        s = re.sub(r'[\r\n\t]', ' ', raw)
        s = re.sub(r'\s{2,}', ' ', s).strip()
        # If ALL CAPS -> Title case
        if s.upper() == s:
            s = " ".join(w.capitalize() for w in re.split(r'\s+', s))
        # remove trailing chars like |,:,• etc
        s = re.sub(r'^[\W_]+|[\W_]+$', '', s)
        # try to use nameparser if available to format (First Last)
        try:
            from nameparser import HumanName
            hn = HumanName(s)
            parts = [p for p in (hn.first, hn.middle, hn.last) if p]
            if parts:
                return " ".join(parts).strip()
        except Exception:
            pass
        return s

    def clean_line(line: str):
        # strip contact noise
        line = re.sub(r'\u2022', ' ', line)  # bullet char
        line = re.sub(r'\s+', ' ', line).strip()
        # remove phone numbers
        line = re.sub(r'(?:(?:\+?\d{1,3}[-.\s]*)?(?:\(?\d{2,4}\)?[-.\s]*)?\d{3,4}[-.\s]*\d{3,4})', '', line)
        # remove urls and emails
        line = re.sub(r'https?://\S+|www\.\S+|\S+@\S+', '', line)
        # remove extra punctuation
        line = re.sub(r'[*_#=~]{2,}', ' ', line)
        return line.strip()

    def scan_top_lines(text):
        lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
        if not lines:
            return None
        SKIP = re.compile(
            r'education|experience|skills|projects|objective|summary|certificat|cgpa|university|college|institute',
            re.I
        )
        # Consider first 12 lines or fewer
        for line in lines[:12]:
            if SKIP.search(line):
                continue
            if re.search(r'\b(email|linkedin|github|http|www\.|@)\b', line, re.I):
                continue
            cleaned = clean_line(line)
            if not cleaned:
                continue
            # ALL CAPS like "SAATWIK DUTTA"
            if re.match(r'^[A-Z\s\-\.\']+$', line) and 1 <= len(cleaned.split()) <= 4:
                return normalize_name(cleaned)
            # Title-like lines: 1-4 words with initial capitals or initials
            tokens = re.findall(r"[A-Za-z][A-Za-z\.\-']+", cleaned)
            if 1 <= len(tokens) <= 4 and not re.search(r'\d', cleaned):
                # prefer those with Capitalized tokens
                cap_tokens = [t for t in tokens if re.match(r'^[A-Z][a-z]', t) or re.match(r'^[A-Z]\.$', t)]
                if cap_tokens:
                    candidate = " ".join(cap_tokens[:4])
                    return normalize_name(candidate)
        return None

    def name_from_email(text):
        em = None
        m = re.search(r'([A-Za-z0-9._%+\-]+)@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})', text)
        if m:
            em = m.group(1)
        if not em:
            return None
        local = re.sub(r'\d+', '', em).lower()
        local = re.sub(r'^(resume|cv|contact)[._\-]?', '', local, flags=re.I)
        # delimiter case
        if re.search(r'[._\-]', local):
            parts = [p for p in re.split(r'[._\-]+', local) if p and len(p) > 1]
            if parts:
                return " ".join(p.capitalize() for p in parts[:3])
        # camelCase split
        if re.search(r'[a-z][A-Z]', em):
            s = re.sub(r'([a-z])([A-Z])', r'\1 \2', em)
            return " ".join(p.capitalize() for p in s.split()[:3])
        # mashed string — try balanced split with heuristics
        if local.isalpha() and len(local) >= 6:
            best = None
            best_score = -999
            L = len(local)
            for i in range(2, L-1):
                first, last = local[:i], local[i:]
                if len(first) < 2 or len(last) < 2:
                    continue
                score = 0
                if len(first) >= 3 and len(last) >= 3:
                    score += 3
                if re.search(r'[aeiou]', first) and re.search(r'[aeiou]', last):
                    score += 1
                score -= abs(len(first)-len(last))*0.1
                if re.match(r'^[bcdfghjklmnpqrstvwxyz]', last):
                    score += 0.5
                if score > best_score:
                    best_score = score
                    best = (first, last)
            if best:
                return f"{best[0].capitalize()} {best[1].capitalize()}"
        # fallback
        return local.capitalize()

    # ---------- execution order ----------
    # 1) spaCy NER
    spacy_name = try_spacy(text)
    if spacy_name and spacy_name.lower() not in ('resume', 'cv', 'unknown'):
        return spacy_name

    # 2) top-line heuristics
    top_name = scan_top_lines(text)
    if top_name and top_name.lower() not in ('resume', 'cv', 'unknown'):
        return top_name

    # 3) email-based
    email_name = name_from_email(text)
    if email_name and email_name.lower() not in ('resume', 'cv', 'unknown'):
        return email_name

    # 4) filename fallback
    if original_filename:
        fn = original_filename.rsplit(".", 1)[0]
        fn = re.sub(r'[_\-\d]+', ' ', fn).strip()
        if fn:
            return " ".join(p.capitalize() for p in fn.split()[:4])

    return "Unknown"


@app.post("/match")
async def match(jd: UploadFile = File(...), resumes: List[UploadFile] = File(...), top_k: int = 5):
    MAX_RESUMES = 20
    if len(resumes) > MAX_RESUMES:
        raise HTTPException(status_code=400, detail=f"Max {MAX_RESUMES} resumes allowed for demo.")

    temp_paths = []

    try:
        jd_path = await save_upload_to_tempfile(jd)
        temp_paths.append(jd_path)
        try:
            jd_text = extract_text_auto(jd_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract JD text: {e}")

        # single resume quick path
        if len(resumes) == 1:
            resume_path = await save_upload_to_tempfile(resumes[0])
            temp_paths.append(resume_path)
            try:
                resume_text = extract_text_auto(resume_path)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to extract resume text: {e}")

            # compute similarity
            score = cosine_similarity_single_resume(jd_text, resume_text)

            # also extract simple metadata + explanationexplanation to be more comprehensive
            name = extract_name_simple(resume_text, resumes[0].filename or "unknown")
            email = extract_email(resume_text)
            # explanation = explain_match_by_sentence(jd_text, resume_text, top_n=2)

            resp = {
            "table": [
                {
                    "Name": name,
                    "Email": email,
                    "Score": float(score),
                }
                ],
            }

            return resp

        # multiple resumes path
        resumes_texts = []
        original_filenames = []
        for r in resumes:
            resume_path = await save_upload_to_tempfile(r)
            temp_paths.append(resume_path)
            try:
                resume_txt = extract_text_auto(resume_path)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to extract resume '{r.filename}': {e}")
            resumes_texts.append(resume_txt)
            original_filenames.append(r.filename or "unknown")

        # ranking
        ranked = rank_resumes(jd_text, resumes_texts, top_k=len(resumes_texts))
        results = []
        for idx, score in ranked[:top_k]:
            txt = resumes_texts[idx]
            name = extract_name_simple(txt, original_filenames[idx])
            email = extract_email(txt)
            # explanation = explain_match_by_sentence(jd_text, txt, top_n=2)
            results.append({
            "resume_index": idx,
            "original_filename": original_filenames[idx],
            "name": name,
            "email": email,
            "score": score,
            # "explanation": explanation
        })
            
        return {
            "mode": "batch",
            "table": results,   # this is now your table
            "count": len(results)
        }

    finally:
        # cleanup all temp files we created
        for p in temp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass

# @app.post("/generate_ics")
# async def generate_ics(summary: str = "Interview", description: str = "", start_iso: str = "", end_iso: str = "", location: str = ""):
#     """
#     Generate an .ics file and return it.
#     Expects simple ISO strings like 2025-11-08T15:00:00 (local naive time).
#     """
#     if not start_iso or not end_iso:
#         raise HTTPException(status_code=400, detail="start_iso and end_iso are required")
#     def iso_to_ics(dt_iso: str):
#         return dt_iso.replace("-", "").replace(":", "")
#     uid = str(uuid.uuid4())
#     ics = f"""BEGIN:VCALENDAR
# VERSION:2.0
# PRODID:-//ResumeMatcherDemo//EN
# BEGIN:VEVENT
# UID:{uid}
# SUMMARY:{summary}
# DTSTART:{iso_to_ics(start_iso)}
# DTEND:{iso_to_ics(end_iso)}
# DESCRIPTION:{description}
# LOCATION:{location}
# END:VEVENT
# END:VCALENDAR
# """
#     tf = tempfile.NamedTemporaryFile(delete=False, suffix=".ics")
#     tf.write(ics.encode("utf-8"))
#     tf.flush()
#     tf.close()
#     return FileResponse(tf.name, filename="invite.ics", media_type="text/calendar")
