

import os
import uuid
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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



def find_model():
    # Check environment variable first
    env_path = os.environ.get("MODEL_PATH")
    if env_path and Path(env_path).exists():
        log.info(f"Using model from environment variable: {env_path}")
        return env_path

    # Check next to this file
    local = Path(__file__).parent / "bart-large-cnn"
    if local.exists():
        log.info(f"Found model next to app.py: {local}")
        return str(local)

    # Check common locations
    common = [
        Path.home() / "programming/quasar/bart-large-cnn",
        Path.home() / "quasar/bart-large-cnn",
        Path("/home/the_programmer/programming/quasar/bart-large-cnn"),
        Path.cwd() / "bart-large-cnn",
    ]
    for p in common:
        if p.exists():
            log.info(f"Found model at: {p}")
            return str(p)

    # Fall back to HuggingFace download
    log.info("Model folder not found locally. Will download sshleifer/distilbart-cnn-12-6")
    log.info("This downloads 600MB on first run, then cached forever.")
    return "sshleifer/distilbart-cnn-12-6"

MODEL_PATH = find_model()
log.info(f"Loading model: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH,
    low_cpu_mem_usage=True,
)
model.eval()
log.info("Model loaded and ready.")

#  Temp directory 
TEMP_DIR  = Path(tempfile.gettempdir()) / "quasar"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
THIS_DIR  = Path(__file__).parent
MAX_MB    = 10
MAX_BYTES = MAX_MB * 1024 * 1024


# Chunking
def chunk_text(text, max_tokens=900):
    words, chunks, current = text.split(), [], []
    for word in words:
        current.append(word)
        if len(current) >= max_tokens:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


# Summarize + build PDF

def run_summary(cleaned_text: str, output_pdf_path: str, original_filename: str):
    chunks = chunk_text(cleaned_text)
    total  = len(chunks)
    final_summary = ""

    log.info(f"Summarizing {total} chunk(s)...")

    for i, chunk in enumerate(chunks):
        log.info(f"Chunk {i+1}/{total}...")

        chunk_word_count = len(chunk.split())

        # Scale tokens dynamically — target ~25% of input (75% reduction)
        dynamic_max_tokens = max(150, int(chunk_word_count * 0.30))
        dynamic_min_tokens = max(80,  int(chunk_word_count * 0.15))

        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        # Use greedy for small chunks, beam for large ones
        is_large_chunk = chunk_word_count > 500

        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=2 if is_large_chunk else 1,   # greedy on small chunks
            max_new_tokens=dynamic_max_tokens,
            min_length=dynamic_min_tokens,
            length_penalty=1.0,
            early_stopping=False,
            no_repeat_ngram_size=3,
        )

        chunk_summary  = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        final_summary += chunk_summary + "\n\n"
        log.info(f"Chunk {i+1}/{total} done.")

    # Build PDF
    doc    = SimpleDocTemplate(
        output_pdf_path, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )
    styles = getSampleStyleSheet()
    story  = []

    story.append(Paragraph("<b>Document Summary</b>", styles["Title"]))
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(f"<i>Source: {original_filename}</i>", styles["Normal"]))
    story.append(Spacer(1, 4*mm))

    orig_w = len(cleaned_text.split())
    summ_w = len(final_summary.split())
    redux  = round((1 - summ_w / max(orig_w, 1)) * 100)
    story.append(Paragraph(
        f"<i>Original: {orig_w:,} words  |  Summary: {summ_w:,} words  |  Reduced: {redux}%</i>",
        styles["Normal"]
    ))
    story.append(Spacer(1, 8*mm))

    for para in final_summary.split("\n\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para, styles["Normal"]))
            story.append(Spacer(1, 4*mm))

    doc.build(story)
    log.info(f"PDF saved: {output_pdf_path}")


#  FastAPI 
app = FastAPI(title="Quasar Summarizer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    app.mount("/static", StaticFiles(directory=str(THIS_DIR)), name="static")
except Exception:
    pass


def delete_files(paths):
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.get("/")
def serve_frontend():
    html_path = THIS_DIR / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return JSONResponse({"error": "index.html not found"})


@app.post("/summarize")
async def summarize(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400,
            detail="Only PDF files accepted.")

    job_id     = str(uuid.uuid4())[:8]
    input_path = str(TEMP_DIR / f"{job_id}_input.pdf")
    out_path   = str(TEMP_DIR / f"{job_id}_summary.pdf")
    log.info(f"[{job_id}] Received: {file.filename}")

    try:
        content = await file.read()
        if len(content) > MAX_BYTES:
            raise HTTPException(status_code=413,
                detail=f"File too large. Max {MAX_MB}MB.")
        Path(input_path).write_bytes(content)

        # Extract
        doc = pymupdf.open(input_path)
        raw_text = ""
        for page in doc:
            raw_text += page.get_text()
            raw_text += "\f"
        doc.close()

        if not raw_text or len(raw_text.strip()) < 50:
            raise HTTPException(status_code=422,
                detail="Could not extract text. May be a scanned PDF.")

        log.info(f"[{job_id}] Extracted {len(raw_text.split()):,} words")

        # Clean
        cleaned = clean_text(raw_text)
        log.info(f"[{job_id}] Cleaned: {len(cleaned.split()):,} words")

        # Summarize
        run_summary(cleaned, out_path, file.filename)

        background_tasks.add_task(delete_files, [input_path])

        stem          = Path(file.filename).stem
        download_name = f"{stem}_summary.pdf"
        log.info(f"[{job_id}] Sending: {download_name}")

        return FileResponse(
            path=out_path,
            media_type="application/pdf",
            filename=download_name,
            headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
        )

    except HTTPException:
        delete_files([input_path, out_path])
        raise
    except Exception as e:
        log.error(f"[{job_id}] Error: {e}", exc_info=True)
        delete_files([input_path, out_path])
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
