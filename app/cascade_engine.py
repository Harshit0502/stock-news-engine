import numpy as np
import joblib
import tensorflow as tf

class CascadeModelEngine:
    def __init__(
        self,
        rnn_p="models/base_rnn.h5",
        gru_p="models/base_gru.h5",
        lstm_p="models/base_lstm.h5",
        meta_p="models/meta_blending.pkl"
    ):
        # Load base models without compilation to avoid custom loss issues
        self.rnn  = tf.keras.models.load_model(rnn_p, compile=False)
        self.gru  = tf.keras.models.load_model(gru_p, compile=False)
        self.lstm = tf.keras.models.load_model(lstm_p, compile=False)
        # Load meta-learner and threshold
        meta_data = joblib.load(meta_p)
        self.meta = meta_data.get('model') if isinstance(meta_data, dict) else meta_data
        self.threshold = meta_data.get('threshold', 0.5) if isinstance(meta_data, dict) else 0.5

    def predict(self, X_seq, neighbor_score=None, return_proba=True):
        """
        Perform cascade prediction.

        Args:
            X_seq (np.ndarray): shape (N, timesteps, features)
            neighbor_score (np.ndarray): optional extra column shape (N,) or (N,1)
            return_proba (bool): if True, return probabilities and binary preds;
                                 else return only preds.

        Returns:
            preds (np.ndarray): binary predictions
            probs (np.ndarray): floating probabilities
        """
        # Base model probabilities
        p_rnn = self.rnn.predict(X_seq, verbose=0).ravel()
        p_gru = self.gru.predict(X_seq, verbose=0).ravel()
        p_lstm = self.lstm.predict(X_seq, verbose=0).ravel()
        # Stack
        p = np.vstack([p_rnn, p_gru, p_lstm]).T
        if neighbor_score is not None:
            # ensure correct shape
            ns = np.array(neighbor_score).reshape(-1,1)
            p = np.concatenate([p, ns], axis=1)
        # Meta probabilities and preds
        probs = self.meta.predict_proba(p)[:,1] if hasattr(self.meta, 'predict_proba') else self.meta.predict(p)
        preds = (probs >= self.threshold).astype(int)
        return (preds, probs) if return_proba else preds
