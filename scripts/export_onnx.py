"""
Convert trained Keras and sklearn models to ONNX for C++ inference.
"""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import os
import joblib
import tensorflow as tf
import tf2onnx
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Paths
KERAS_MODELS = {
    'rnn': 'models/base_rnn.h5',
    'gru': 'models/base_gru.h5',
    'lstm': 'models/base_lstm.h5'
}
META_PATH = 'models/meta_blending.pkl'
ONNX_DIR = 'models/onnx'

# Create output directory
os.makedirs(ONNX_DIR, exist_ok=True)

# Convert each Keras model to ONNX
for name, model_path in KERAS_MODELS.items():
    print(f"Converting Keras model '{name}' to ONNX...")
    # Load the original sequential model
    seq_model = tf.keras.models.load_model(model_path, compile=False)

    # First, wrap the sequential model in a Functional Model to ensure it has
    # the proper structure with defined inputs/outputs. This fixes earlier errors.
    input_tensor = seq_model.inputs[0]
    output_tensor = seq_model(input_tensor)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    
    # ==================== MODIFICATION START (v4) ====================
    #
    # FIX: The error "AttributeError: ... no attribute '_get_save_spec'" is caused by
    # a version mismatch between tf2onnx and a newer version of TensorFlow.
    # We fix this by manually creating the input signature and passing it to the converter.
    # This stops tf2onnx from calling the internal TensorFlow function that no longer exists.
    #
    # We create a TensorSpec tuple from the model's input tensor.
    input_signature = (tf.TensorSpec(input_tensor.shape, input_tensor.dtype, name=input_tensor.name),)
    #
    # ===================== MODIFICATION END (v4) =====================

    # Perform conversion with the explicit input_signature.
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature, # Pass the manually created signature here
        opset=13
    )

    # Save ONNX model to file
    output_file = os.path.join(ONNX_DIR, f"base_{name}.onnx")
    onnx.save_model(onnx_model, output_file)
    print(f"Saved ONNX model to {output_file}")

# Convert meta-learner (sklearn) to ONNX
print("Converting meta-learner to ONNX...")
meta_data = joblib.load(META_PATH)
meta_model = meta_data['model'] if isinstance(meta_data, dict) else meta_data

# Dynamically check the number of features the model expects
try:
    n_features = meta_model.n_features_in_
    print(f"Meta-learner expects {n_features} input features.")
except AttributeError:
    n_features = 3
    print("Warning: Could not determine number of input features for meta-learner. Defaulting to 3.")

initial_type = [('float_input', FloatTensorType([None, n_features]))]
onx_meta = convert_sklearn(meta_model, initial_types=initial_type)
meta_file = os.path.join(ONNX_DIR, 'meta_model.onnx')
with open(meta_file, 'wb') as f:
    f.write(onx_meta.SerializeToString())
print(f"Saved meta-learner ONNX to {meta_file}")