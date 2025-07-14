from fusion_cpp import knn  # pybind11
neighbor = knn(latest_emb.astype('float32'), 5)[1].mean()  # avg dist or label
final_pred, prob = engine.predict(X_seq=np.array([seq]), neighbor_score=np.array([[neighbor]]))
