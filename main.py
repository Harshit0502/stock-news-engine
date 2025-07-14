from app.cascade_engine import CascadeModelEngine
engine = CascadeModelEngine()
pred, prob = engine.predict(X_seq_batch)
