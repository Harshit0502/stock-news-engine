import numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression

p_rnn  = np.loadtxt("data/preds_rnn.csv", delimiter=",")
p_gru  = np.loadtxt("data/preds_gru.csv", delimiter=",")
p_lstm = np.loadtxt("data/preds_lstm.csv", delimiter=",")
y = pd.read_csv("data/processed_dataset_labeled.csv")['label'].values[5:]   # align window shift

X_meta = np.column_stack([p_rnn, p_gru, p_lstm])  # add neighbor_score later
meta = LogisticRegression(max_iter=1000)
meta.fit(X_meta, y)
joblib.dump(meta, "models/meta_blending.pkl")
