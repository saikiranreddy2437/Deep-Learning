import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, outputs):
        weights = torch.softmax(self.attn(outputs), dim=1)
        context = torch.sum(weights * outputs, dim=1)
        return context

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, inputs):
        x = self.embedding(inputs)
        outputs, _ = self.lstm(x)
        context = self.attention(outputs)
        return self.fc(context)
