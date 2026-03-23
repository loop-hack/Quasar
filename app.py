"""
Quasar Summarizer

WHAT THIS FILE DOES:
    Receives a PDF from the browser
    calls clean_text() from your data_processing.py
    calls run_summary() defined here (uses your summery_pdf.py logic)
     returns the summary PDF for download

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
from data_processing import clean_text   #existing function

MODEL_PATH = "sshleifer/distilbart-cnn-12-6"



from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

log = logging.getLogger(__name__)
log.info(f"Loading model from: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

log.info("Model loaded.")


def chunk_text(text, max_tokens=900):
    """
    Exact copy of your chunk_text() from summery_pdf.py.
    Splits text into chunks of max_tokens words.
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


def run_summary(cleaned_text: str, output_pdf_path: str, original_filename: str):
    """
    Exact logic from your summery_pdf.py, made into a function.
    Takes cleaned text string, writes summary PDF to output_pdf_path.
    """
    chunks        = chunk_text(cleaned_text)
    final_summary = ""

    for i, chunk in enumerate(chunks):
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=200,
            min_length=40,
            length_penalty=2.0,
            early_stopping=True,
        )

        chunk_summary  = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        final_summary += chunk_summary + "\n\n"
        log.info(f"Chunk {i+1}/{len(chunks)} summarized.")

    # Build the PDF  same as summery_pdf.py
    doc    = SimpleDocTemplate(output_pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story  = []

    story.append(Paragraph("<b>Document Summary</b>", styles["Title"]))
    story.append(Paragraph(f"<i>Source: {original_filename}</i>", styles["Normal"]))
    story.append(Paragraph("&nbsp;", styles["Normal"]))
    story.append(Paragraph(
        final_summary.replace("\n", "<br/>"),
        styles["Normal"]
    ))

    doc.build(story)
    log.info(f"Summary PDF saved: {output_pdf_path}")


#  FASTAPI APP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
)

app = FastAPI(title="Quasar Summarizer", version="1.0.0")

# Allow the browser to call this backend from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html as the root page
# Make sure index.html is in the same folder as app.py
THIS_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(THIS_DIR)), name="static")

TEMP_DIR  = Path(tempfile.gettempdir()) / "quasar"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

MAX_MB    = 10
MAX_BYTES = MAX_MB * 1024 * 1024


def delete_files(paths):
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass

#health check
@app.get("/health")
def health():
    return {"status": "ok"}


#  Serve frontend 
@app.get("/")
def serve_frontend():
    html_path = THIS_DIR / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return {"error": "index.html not found in the same folder as app.py"}


#  Main summarize endpoint 
@app.post("/summarize")
async def summarize(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Pipeline:
        PDF upload >>>
        pymupdf extracts raw text>>
        clean_text() from your data_processing.py>>
        run_summary() — your summery_pdf.py logic, made dynamic>>
        FileResponse — user downloads the summary PDF
    """

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
        #  Save uploaded PDF 
        content = await file.read()
        if len(content) > MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max {MAX_MB}MB allowed.",
            )
        Path(input_path).write_bytes(content)
        log.info(f"[{job_id}] Saved ({len(content)//1024} KB)")

        #   Extract raw text from PDF 
        # Using pymupdf directly same as your data_processing.py does
        log.info(f"[{job_id}] Extracting text from PDF...")
        doc      = pymupdf.open(input_path)
        raw_text = ""
        for page in doc:
            raw_text += page.get_text()
            raw_text += "\f"   # page break  same as your read_pdf()
        doc.close()

        if not raw_text or len(raw_text.strip()) < 50:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Could not extract text from this PDF. "
                    "It may be a scanned/image-only document."
                ),
            )
        log.info(f"[{job_id}] Extracted {len(raw_text.split())} words")

        #  Clean text using YOUR function 
        # clean_text() is imported directly from your data_processing.py
        # Nothing changed in your file.
        log.info(f"[{job_id}] Cleaning text...")
        cleaned = clean_text(raw_text)
        log.info(f"[{job_id}] Cleaned: {len(cleaned.split())} words")

        #  Summarize using YOUR logic 
        # run_summary() is the exact logic from your summery_pdf.py
        # but accepts text as input instead of reading a hardcoded file.
        log.info(f"[{job_id}] Summarizing (30-120s on CPU)...")
        run_summary(cleaned, out_path, file.filename)

        #  Cleanup temp files after download 
        background_tasks.add_task(delete_files, [input_path, out_path])

        #  Return summary PDF as download 
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
