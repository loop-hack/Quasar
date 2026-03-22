from transformers import AutoModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


model = AutoModel.from_pretrained("bert-base-cased")


save_path = "./bart-large-cnn"

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)