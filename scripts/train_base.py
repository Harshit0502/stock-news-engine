import pandas as pd, numpy as np, tensorflow as tf
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from utils.seq_windows import make_windows
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Dense, Dropout

df = pd.read_csv("data/processed_dataset_labeled.csv")
y = df['label'].values
X = df.drop(columns=['label','Date']).values

# 1. scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. make sequences
X_seq, y_seq = make_windows(X_scaled, y, n_steps=5)

Xtr, Xval, ytr, yval = train_test_split(X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42)

def build(model_type):
    net = Sequential()
    if model_type=="rnn":
        net.add(SimpleRNN(64, input_shape=Xtr.shape[1:], activation='tanh'))
    elif model_type=="gru":
        net.add(GRU(64, input_shape=Xtr.shape[1:], activation='tanh'))
    else:
        net.add(LSTM(64, return_sequences=False, input_shape=Xtr.shape[1:], activation='tanh'))
    net.add(Dense(1, activation='sigmoid'))
    net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return net

for name in ['rnn','gru','lstm']:
    model = build(name)
    model.fit(Xtr, ytr, epochs=20, validation_data=(Xval,yval),
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    model.save(f"models/base_{name}.h5")
    np.savetxt(f"data/preds_{name}.csv", model.predict(X_seq), delimiter=",")
