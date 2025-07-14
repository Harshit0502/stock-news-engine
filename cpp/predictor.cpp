#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <string>
#include <memory> // Required for std::unique_ptr

// Use the C++ ONNX Runtime API
#include <onnxruntime_cxx_api.h>

// Helper function to read a CSV file into a 2D float vector, skipping the header.
std::vector<std::vector<float>> read_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Error: Could not open CSV file " << path << std::endl;
        exit(1);
    }
    std::string line;
    std::vector<std::vector<float>> data;
    std::getline(in, line); // Skip header row
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;
        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stof(cell));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Warning: Could not convert '" << cell << "' to float. Skipping row." << std::endl;
                row.clear();
                break;
            }
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    return data;
}

// Helper function to create sliding windows of data for sequence models.
std::vector<std::vector<float>> make_windows(const std::vector<std::vector<float>>& X, int n_steps) {
    std::vector<std::vector<float>> windows;
    if (X.size() <= n_steps) {
        std::cerr << "Warning: Not enough data points to create a window of size " << n_steps << std::endl;
        return windows;
    }
    for (size_t i = n_steps; i < X.size(); ++i) {
        std::vector<float> seq;
        for (size_t j = i - n_steps; j < i; ++j) {
            seq.insert(seq.end(), X[j].begin(), X[j].end());
        }
        windows.push_back(seq);
    }
    return windows;
}

// A generic function to run inference on a session.
// It takes the session, input/output names, and the input tensor, and returns the output tensor.
Ort::Value run_inference(Ort::Session& sess, const char* input_name, const char* output_name, const Ort::Value& input_tensor) {
    // The C API requires separate char* arrays for input and output names.
    const char* input_names[] = {input_name};
    const char* output_names[] = {output_name};
    auto output_tensors = sess.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    return std::move(output_tensors.front());
}


int main() {
    // ======== 1. Initialization ========
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cascade-inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); // Enable all optimizations

    // ======== 2. Load Models and Get I/O Names (ONCE) ========
    // Keras models typically name the input 'input_1' or similar and outputs by the last layer name.
    // skl2onnx names the input 'float_input' by default.
    // NOTE: You can verify these names by opening the .onnx file in a viewer like Netron.
    const char* rnn_input_name = "input_1"; // Default name for the first input layer in Keras
    const char* rnn_output_name = "dense_1"; // Default name for a dense output layer

    const char* gru_input_name = "input_2";
    const char* gru_output_name = "dense_3";

    const char* lstm_input_name = "input_3";
    const char* lstm_output_name = "dense_5";
    
    const char* meta_input_name = "float_input"; // Default skl2onnx input name
    const char* meta_output_name_prob = "probabilities"; // skl2onnx gives this name for probabilities
    const char* meta_output_name_label = "label"; // and this name for the final class label


    Ort::Session rnn_sess(env, "models/onnx/base_rnn.onnx", session_options);
    Ort::Session gru_sess(env, "models/onnx/base_gru.onnx", session_options);
    Ort::Session lstm_sess(env, "models/onnx/base_lstm.onnx", session_options);
    Ort::Session meta_sess(env, "models/onnx/meta_model.onnx", session_options);
    

    // ======== 3. Prepare Data ========
    const int N_STEPS = 5;
    auto raw_data = read_csv("data/processed_dataset_labeled.csv");
    auto windows = make_windows(raw_data, N_STEPS);
    if (windows.empty()) {
        std::cerr << "No windows were created from the data. Exiting." << std::endl;
        return 1;
    }

    // Get the number of features from the first data row.
    const size_t feature_count = raw_data[0].size();
    
    std::ofstream out_file("data/cpp_preds.csv");
    out_file << "pred,prob\n";

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // ======== 4. Main Inference Loop ========
    for (const auto& sequence : windows) {
        // --- Base Model Inference ---
        // Define the shape for the Keras models: [batch_size, n_steps, n_features]
        std::array<int64_t, 3> input_shape = {1, N_STEPS, static_cast<int64_t>(feature_count)};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(sequence.data()), sequence.size(), input_shape.data(), input_shape.size());

        // Run the 3 base models
        Ort::Value rnn_out_tensor = run_inference(rnn_sess, rnn_input_name, rnn_output_name, input_tensor);
        Ort::Value gru_out_tensor = run_inference(gru_sess, gru_input_name, gru_output_name, input_tensor);
        Ort::Value lstm_out_tensor = run_inference(lstm_sess, lstm_input_name, lstm_output_name, input_tensor);

        float p_rnn = rnn_out_tensor.GetTensorMutableData<float>()[0];
        float p_gru = gru_out_tensor.GetTensorMutableData<float>()[0];
        float p_lstm = lstm_out_tensor.GetTensorMutableData<float>()[0];

        // --- Meta-Learner Inference ---
        std::vector<float> meta_input_data = {p_rnn, p_gru, p_lstm};
        std::array<int64_t, 2> meta_input_shape = {1, 3}; // Shape: [batch_size, n_features]
        Ort::Value meta_input_tensor = Ort::Value::CreateTensor<float>(memory_info, meta_input_data.data(), meta_input_data.size(), meta_input_shape.data(), meta_input_shape.size());
        
        // For the scikit-learn model, we ask for both the label and the probability.
        const char* meta_output_names[] = {meta_output_name_label, meta_output_name_prob};
        auto meta_output_tensors = meta_sess.Run(Ort::RunOptions{nullptr}, &meta_input_name, &meta_input_tensor, 1, meta_output_names, 2);

        int64_t final_prediction = meta_output_tensors[0].GetTensorMutableData<int64_t>()[0];
        float final_probability = meta_output_tensors[1].GetTensorMutableData<float>()[1]; // Prob of class '1'

        out_file << final_prediction << "," << final_probability << "\n";
    }

    out_file.close();
    std::cout << "C++ cascade predictions saved to data/cpp_preds.csv" << std::endl;

    return 0;
}