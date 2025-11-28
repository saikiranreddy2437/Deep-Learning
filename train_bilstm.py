import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import argparse
from model_bilstm import BiLSTM
from utils import build_vocab, encode_text

parser = argparse.ArgumentParser()
parser.add_argument("--train")
parser.add_argument("--val")
parser.add_argument("--embedding_dim", default=100, type=int)
parser.add_argument("--hidden_dim", default=128, type=int)
parser.add_argument("--epochs", default=10, type=int)
args = parser.parse_args()

df_train = pd.read_csv(args.train, sep="\t")
df_val = pd.read_csv(args.val, sep="\t")

train_texts = (df_train.turn1 + " [SEP] " + df_train.turn2 + " [SEP] " + df_train.turn3).tolist()
val_texts = (df_val.turn1 + " [SEP] " + df_val.turn2 + " [SEP] " + df_val.turn3).tolist()

vocab = build_vocab(train_texts)

class EmoDataset(Dataset):
    def __init__(self, texts, labels):
        self.x = [encode_text(t, vocab) for t in texts]
        self.y = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])

train_ds = EmoDataset(train_texts, df_train.label.values)
val_ds = EmoDataset(val_texts, df_val.label.values)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

model = BiLSTM(len(vocab), args.embedding_dim, args.hidden_dim, num_labels=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete.")

torch.save(model.state_dict(), "outputs/bilstm_best.pt")
print("BiLSTM model saved!")
