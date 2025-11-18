import tempfile
import re
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from textextract_utils import extract_text_auto
from embed_utils import embed_text
from matcher import rank_resumes, explain_match_by_sentence
from typing import List

app=FastAPI()

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

def extract_name_simple(text:str, original_filename: str = None):
    pass


def extract_email(text:str):
    m=EMAIL_RE.search(text)
    return m.group(0) if m else None


async def save_upload_to_tempfile(upload: UploadFile) -> str:
    filename= (upload.filename or "").lower()
    suffix=''
    if filename.endswith('pdf'):
        suffix = ".pdf"
    elif filename.endswith('docx'):
        suffix = ".docx"
    if filename.endswith('txt'):
        suffix = ".txt"

    tf=tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    data=await upload.read()
    tf.write(data)
    tf.flush()
    tf.close()
    
    return tf.name

@app.post('/match')
async def match (jd: UploadFile=File(...), resumes: List[UploadFile]=File(...), top_k: int=5):
    MAX_RESUMES=20
    if len(resumes)>MAX_RESUMES:
        raise HTTPException(status_code=400, detail=f'maximumn of {MAX_RESUMES} allowed')
    
    jd_path= await save_upload_to_tempfile(jd)
    jd_text= extract_text_auto(jd_path)

    resume_texts=[]
    original_filenames= []

    for r in resumes:
        p=await save_upload_to_tempfile(r)
        txt= extract_text_auto(p)
        resume_texts.append(txt)
        original_filenames.append(r.filename or "unknown")

    ranking= rank_resumes(jd_text, resume_texts, top_k=len(resumes))

    results=[]

    for idx, score in ranking[:top_k]:
        txt=resume_texts[idx]
        name=extract_name_simple(txt, original_filenames[idx])
        email=extract_email(txt)
        explanation = explain_match_by_sentence (jd_text, resume_texts, top_n=2)

        results.append({
            "resume_index": idx,
            "original_filename": original_filenames[idx],
            "name": name,
            "email": email,
            "score": score,
            "explanation": explanation
        }
    )
        
    return JSONResponse({"Results: ": results, "Count: ":len(resumes)})