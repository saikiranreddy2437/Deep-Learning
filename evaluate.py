import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score, classification_report

model_dir = "outputs/bert"
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)

df = pd.read_csv("data/val.tsv", sep="\t")
texts = (df.turn1 + " [SEP] " + df.turn2 + " [SEP] " + df.turn3).tolist()
labels = df.label.values

enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

outputs = model(**enc)
preds = outputs.logits.argmax(dim=1).numpy()

print("Macro F1:", f1_score(labels, preds, average='macro'))
print(classification_report(labels, preds))
