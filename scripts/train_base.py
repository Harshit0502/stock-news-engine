import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from utils.seq_windows import make_windows

# 1. Load data
df = pd.read_csv("data/processed_dataset_labeled.csv")
y = df['label'].values
X = df.drop(columns=['label','Date']).values

# 2. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Windows
X_seq, y_seq = make_windows(X_scaled, y, n_steps=5)

# 4. Train/test split
Xtr, Xval, ytr, yval = train_test_split(
    X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42
)

# 5. Compute class weights
cw = class_weight.compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
class_weights = {i: cw[i] for i in range(len(cw))}

# 6. Optional: define focal loss
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt   = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return alpha * tf.pow(1-pt, gamma) * bce
    return loss

# 7. Model builder
from tensorflow.keras.layers import (
    SimpleRNN, GRU, LSTM, Dense, Dropout, BatchNormalization
)
def build_model(kind):
    inp_shape = Xtr.shape[1:]
    net = tf.keras.Sequential()
    if kind == 'rnn':
        net.add(SimpleRNN(128, activation='tanh', return_sequences=True, input_shape=inp_shape))
        net.add(Dropout(0.2))
        net.add(SimpleRNN(64, activation='tanh'))
    elif kind == 'gru':
        net.add(GRU(128, activation='tanh', return_sequences=True, input_shape=inp_shape))
        net.add(Dropout(0.2))
        net.add(GRU(64, activation='tanh'))
    else:
        net.add(LSTM(128, return_sequences=True, activation='tanh', input_shape=inp_shape))
        net.add(BatchNormalization())
        net.add(Dropout(0.2))
        net.add(LSTM(64, activation='tanh'))
    net.add(Dense(32, activation='elu'))
    net.add(Dropout(0.3))
    net.add(Dense(1, activation='sigmoid'))
    net.compile(
        optimizer='adam',
        loss=focal_loss(gamma=2., alpha=.25),
        metrics=['accuracy']
    )
    return net

# 8. Train and save, plus export predictions
for kind in ['rnn','gru','lstm']:
    model = build_model(kind)
    model.fit(
        Xtr, ytr,
        validation_data=(Xval, yval),
        epochs=30,
        batch_size=64,
        class_weight=class_weights,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )
    model.save(f"models/base_{kind}.h5")
    # Export raw probabilities for entire sequence set
    probs = model.predict(X_seq, verbose=0).ravel()
    np.savetxt(f"data/preds_{kind}.csv", probs, delimiter=",")
    print(f"[{kind}] Mean prob: {probs.mean():.3f}, Std: {probs.std():.3f}")
