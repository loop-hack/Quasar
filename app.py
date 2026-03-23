"""
app.py  -  Quasar Summarizer Backend
=====================================
Pipeline:
    PDF upload
    -> pymupdf extracts raw text
    -> clean_text() from data_processing.py
    -> run_summary() uses LongT5 model
    -> returns summary PDF for download

HOW TO RUN:
    source venv/bin/activate
    uvicorn app:app --reload --port 8000

Open browser: http://localhost:5500  (run python -m http.server 5500 in another terminal)
"""

import os
import uuid
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import pymupdf
from data_processing import clean_text

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

#  Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
)
log = logging.getLogger(__name__)

#  Model 
# LongT5 handles 16,384 tokens in one pass (~12,000 words)
# No chunking needed for most documents
# 800MB size, faster than bart-large on CPU
MODEL_PATH = "pszemraj/long-t5-tglobal-base-16384"

log.info(f"Loading model: {MODEL_PATH}")
log.info("This takes 2-3 minutes on first run (downloading 800MB)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH,
    low_cpu_mem_usage=True,
)
model.eval()

log.info("Model loaded and ready.")

#  Temp directory 
TEMP_DIR = Path(tempfile.gettempdir()) / "quasar"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

MAX_MB    = 10
MAX_BYTES = MAX_MB * 1024 * 1024

THIS_DIR  = Path(__file__).parent


#  Text chunking 
def chunk_text(text, max_tokens=10000):
    """
    Split text into chunks of max_tokens words.
    LongT5 handles 16,384 tokens so most documents
    fit in a single chunk — no quality loss from splitting.
    """
    words   = text.split()
    chunks  = []
    current = []

    for word in words:
        current.append(word)
        if len(current) >= max_tokens:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


#  Summarization + PDF creation 
def run_summary(cleaned_text: str, output_pdf_path: str, original_filename: str):
    """
    Summarize cleaned text using LongT5 and write output PDF.

    LongT5 improvements over BART:
    - Reads up to 16,384 tokens at once (vs 1024 for BART)
    - More detailed summaries (max_new_tokens=500)
    - No repeated sentences (no_repeat_ngram_size=3)
    - Faster on CPU (num_beams=2)
    """
    chunks        = chunk_text(cleaned_text)
    final_summary = ""
    total         = len(chunks)

    log.info(f"Summarizing {total} chunk(s)...")

    for i, chunk in enumerate(chunks):
        log.info(f"Processing chunk {i+1}/{total}...")

        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=16384,          # LongT5 full context window
        )

        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=2,               # faster on CPU, still good quality
            max_new_tokens=500,        # detailed summary per chunk
            min_length=100,            # never too short
            length_penalty=2.0,        # rewards longer summaries
            early_stopping=True,
            no_repeat_ngram_size=3,    # prevents repetition
        )

        chunk_summary  = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        final_summary += chunk_summary + "\n\n"
        log.info(f"Chunk {i+1}/{total} done.")

    log.info("All chunks done. Building PDF...")

    #  Build PDF 
    doc    = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )
    styles = getSampleStyleSheet()
    story  = []

    # Title
    story.append(Paragraph("<b>Document Summary</b>", styles["Title"]))
    story.append(Spacer(1, 6*mm))

    # Source file
    story.append(Paragraph(
        f"<i>Source: {original_filename}</i>",
        styles["Normal"]
    ))
    story.append(Spacer(1, 4*mm))

    # Stats line
    original_words = len(cleaned_text.split())
    summary_words  = len(final_summary.split())
    reduction      = round((1 - summary_words / max(original_words, 1)) * 100)
    story.append(Paragraph(
        f"<i>Original: {original_words:,} words &nbsp;|&nbsp; "
        f"Summary: {summary_words:,} words &nbsp;|&nbsp; "
        f"Reduced by: {reduction}%</i>",
        styles["Normal"]
    ))
    story.append(Spacer(1, 8*mm))

    # Summary paragraphs
    for para in final_summary.split("\n\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para, styles["Normal"]))
            story.append(Spacer(1, 4*mm))

    doc.build(story)
    log.info(f"PDF saved: {output_pdf_path}")


#  FastAPI app 
app = FastAPI(title="Quasar Summarizer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(THIS_DIR)), name="static")


def delete_files(paths):
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


#  Health check 
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


#  Serve frontend 
@app.get("/")
def serve_frontend():
    html_path = THIS_DIR / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return {"error": "index.html not found"}


#  Summarize 
@app.post("/summarize")
async def summarize(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files accepted. Please upload a .pdf file.",
        )

    job_id     = str(uuid.uuid4())[:8]
    input_path = str(TEMP_DIR / f"{job_id}_input.pdf")
    out_path   = str(TEMP_DIR / f"{job_id}_summary.pdf")

    log.info(f"[{job_id}] Received: {file.filename}")

    try:
        #  Save uploaded file 
        content = await file.read()
        if len(content) > MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max {MAX_MB}MB allowed.",
            )
        Path(input_path).write_bytes(content)
        log.info(f"[{job_id}] Saved ({len(content)//1024}KB)")

        #  Extract text 
        log.info(f"[{job_id}] Extracting text...")
        doc      = pymupdf.open(input_path)
        raw_text = ""
        for page in doc:
            raw_text += page.get_text()
            raw_text += "\f"
        doc.close()

        if not raw_text or len(raw_text.strip()) < 50:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Could not extract text from this PDF. "
                    "It may be a scanned or image-only document."
                ),
            )
        log.info(f"[{job_id}] Extracted {len(raw_text.split()):,} words")

        #  Clean text 
        log.info(f"[{job_id}] Cleaning text...")
        cleaned = clean_text(raw_text)
        log.info(f"[{job_id}] Cleaned: {len(cleaned.split()):,} words")

        #  Summarize 
        log.info(f"[{job_id}] Summarizing...")
        run_summary(cleaned, out_path, file.filename)

        #  Return PDF 
        background_tasks.add_task(delete_files, [input_path, out_path])

        stem          = Path(file.filename).stem
        download_name = f"{stem}_summary.pdf"

        log.info(f"[{job_id}] Done. Sending: {download_name}")

        return FileResponse(
            path=out_path,
            media_type="application/pdf",
            filename=download_name,
            headers={
                "Content-Disposition": f'attachment; filename="{download_name}"'
            },
        )

    except HTTPException:
        delete_files([input_path, out_path])
        raise
    except Exception as e:
        log.error(f"[{job_id}] Error: {e}", exc_info=True)
        delete_files([input_path, out_path])
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}",
        )