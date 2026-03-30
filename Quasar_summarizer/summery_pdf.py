from transformers import AutoModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4




model_path = "/home/the_programmer/programming/quasar/bart-large-cnn"   #local model folder

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

with open("/home/the_programmer/programming/quasar/frontend/output5.txt", "r", encoding="utf-8") as f:
    text = f.read()

def chunk_text(text, max_tokens=900):
    words = text.split()
    chunks = []
    current = []

    for word in words:
        current.append(word)
        if len(current) >= max_tokens:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks

chunks = chunk_text(text)

final_summary = ""

for i, chunk in enumerate(chunks):
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=200,
        min_length=40,
        length_penalty=2.0,
        early_stopping=True
    )

    chunk_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print(f"Summary {i+1} done.\n")
    final_summary += chunk_summary + "\n\n"


doc = SimpleDocTemplate("summary_report1.pdf", pagesize=A4)

styles = getSampleStyleSheet()
style = styles["Normal"]
story = []

story.append(Paragraph("<b>Document Summary</b>", styles['Title']))
story.append(Paragraph(final_summary.replace("\n", "<br/>"), style))

doc.build(story)

print("PDF saved as summary_report.pdf")