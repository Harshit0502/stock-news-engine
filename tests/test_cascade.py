import numpy as np, pandas as pd

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from app.cascade_engine import CascadeModelEngine
from utils.seq_windows import make_windows
# load processed dataset & build a small test batch
df = pd.read_csv("data/processed_dataset_labeled.csv")
X = df.drop(columns=["label", "Date"]).values
y = df["label"].values
X_seq, _ = make_windows(X, y, n_steps=5)

engine = CascadeModelEngine()          # loads saved models
pred, prob = engine.predict(X_seq[:10])  # infer on first 10 windows
print("pred:", pred, "prob:", prob[:3])
