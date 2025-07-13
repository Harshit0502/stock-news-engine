import numpy as np, joblib, tensorflow as tf

class CascadeModelEngine:
    def __init__(self, rnn_p="models/base_rnn.h5", gru_p="models/base_gru.h5",
                 lstm_p="models/base_lstm.h5", meta_p="models/meta_blending.pkl"):
        self.rnn  = tf.keras.models.load_model(rnn_p)
        self.gru  = tf.keras.models.load_model(gru_p)
        self.lstm = tf.keras.models.load_model(lstm_p)
        self.meta = joblib.load(meta_p)

    def predict(self, X_seq, neighbor_score=None):
        """X_seq shape: (N, 5, F)"""
        p = np.column_stack([
            self.rnn.predict(X_seq, verbose=0).ravel(),
            self.gru.predict(X_seq, verbose=0).ravel(),
            self.lstm.predict(X_seq, verbose=0).ravel()
        ])
        if neighbor_score is not None:
            p = np.column_stack([p, neighbor_score])
        final = self.meta.predict(p)
        return final, self.meta.predict_proba(p)[:,1]
