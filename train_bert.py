from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd

train = pd.read_csv("data/train.tsv", sep="\t")
val = pd.read_csv("data/val.tsv", sep="\t")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode(df):
    return tokenizer(
        (df.turn1 + " [SEP] " + df.turn2 + " [SEP] " + df.turn3).tolist(),
        padding=True,
        truncation=True,
    )

train_enc = encode(train)
val_enc = encode(val)

train_labels = train.label.tolist()
val_labels = val.label.tolist()

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

args = TrainingArguments(
    output_dir="outputs/bert",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_enc,
    eval_dataset=val_enc,
)

trainer.train()
trainer.save_model("outputs/bert")
