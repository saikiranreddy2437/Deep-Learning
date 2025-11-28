import re

def build_vocab(texts):
    vocab = {"<PAD>": 0}
    idx = 1
    for t in texts:
        for w in t.lower().split():
            if w not in vocab:
                vocab[w] = idx
                idx += 1
    return vocab

def encode_text(text, vocab, max_len=50):
    words = text.lower().split()
    encoded = [vocab.get(w, 0) for w in words[:max_len]]
    encoded += [0] * (max_len - len(encoded))
    return encoded
