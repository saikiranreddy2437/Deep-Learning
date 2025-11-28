import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--train_out")
parser.add_argument("--val_out")
args = parser.parse_args()

df = pd.read_csv(args.input, sep="\t")

train = df.sample(frac=0.85, random_state=42)
val = df.drop(train.index)

train.to_csv(args.train_out, sep="\t", index=False)
val.to_csv(args.val_out, sep="\t", index=False)

print("Train/Val Split saved!")
