# scripts/day2_blend_meta.py

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve

# 1. Load base-model probs
p_rnn  = np.loadtxt("data/preds_rnn.csv", delimiter=",")
p_gru  = np.loadtxt("data/preds_gru.csv", delimiter=",")
p_lstm = np.loadtxt("data/preds_lstm.csv", delimiter=",")
X_meta = np.vstack([p_rnn, p_gru, p_lstm]).T

# 2. Load labels (aligned to window)
y = pd.read_csv("data/processed_dataset_labeled.csv")['label'].values[5:]

# 3. Train/test split for meta (optional, here we use full data)
# from sklearn.model_selection import train_test_split
# Xm_tr, Xm_val, y_tr, y_val = train_test_split(...)

# 4. Fit a tree-based meta-learner
meta = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
meta.fit(X_meta, y)

# 5. Optimize F1 threshold on OOB or holdout
probs_meta = meta.predict_proba(X_meta)[:,1]
prec, rec, thresh = precision_recall_curve(y, probs_meta)
f1_scores = 2*prec*rec/(prec+rec+1e-9)
best_t = thresh[f1_scores.argmax()]
print(f"Best meta threshold: {best_t:.3f}, Max F1: {f1_scores.max():.3f}")

# 6. Save model + threshold
joblib.dump({'model': meta, 'threshold': best_t}, "models/meta_blending.pkl")
