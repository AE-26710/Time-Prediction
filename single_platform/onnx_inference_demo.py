"""
ONNX Model Inference Example

This script demonstrates how to use the exported ONNX models for prediction.
"""

import numpy as np
import onnxruntime as ort
import os

def load_onnx_model(model_path):
    """Load ONNX model and return inference session"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    session = ort.InferenceSession(model_path)
    return session

def predict_single(session, input_size):
    """
    Make prediction using ONNX model
    
    Args:
        session: ONNX Runtime inference session
        input_size: Single input value or list of input values
    
    Returns:
        Predicted runtime(s) in milliseconds
    """
    # Convert to numpy array with correct shape
    if isinstance(input_size, (int, float)):
        input_data = np.array([[input_size]], dtype=np.float32)
    else:
        input_data = np.array([[x] for x in input_size], dtype=np.float32)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(None, {input_name: input_data})
    
    # Return predictions
    predictions = outputs[0].flatten()
    return predictions[0] if len(predictions) == 1 else predictions

def predict_batch(session, input_sizes):
    """
    Make batch predictions using ONNX model
    
    Args:
        session: ONNX Runtime inference session
        input_sizes: List of input values
    
    Returns:
        Array of predicted runtimes in milliseconds
    """
    input_data = np.array([[x] for x in input_sizes], dtype=np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    return outputs[0].flatten()

def compare_models(best_model_path, ensemble_model_path, test_inputs):
    """
    Compare predictions from best single model and ensemble model
    """
    print("="*80)
    print("Comparing Best Single Model vs Ensemble Model")
    print("="*80)
    
    # Load models
    best_session = load_onnx_model(best_model_path)
    ensemble_session = load_onnx_model(ensemble_model_path)
    
    # Make predictions
    print(f"\n{'Input Size':<15} {'Best Single':<15} {'Ensemble':<15} {'Difference':<15}")
    print("-"*60)
    
    for input_size in test_inputs:
        best_pred = predict_single(best_session, input_size)
        ensemble_pred = predict_single(ensemble_session, input_size)
        diff = abs(best_pred - ensemble_pred)
        
        print(f"{input_size:<15.0f} {best_pred:<15.4f} {ensemble_pred:<15.4f} {diff:<15.4f}")
    
    print("="*80)

def benchmark_inference_speed(model_path, n_runs=1000):
    """
    Benchmark inference speed of ONNX model
    """
    import time
    
    session = load_onnx_model(model_path)
    input_data = np.array([[1024]], dtype=np.float32)
    input_name = session.get_inputs()[0].name
    
    # Warm-up
    for _ in range(10):
        session.run(None, {input_name: input_data})
    
    # Benchmark
    start_time = time.time()
    for _ in range(n_runs):
        session.run(None, {input_name: input_data})
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / n_runs * 1000
    
    print(f"\nInference Speed Benchmark ({n_runs} runs):")
    print(f"  Average inference time: {avg_time_ms:.4f} ms")
    print(f"  Throughput: {1000/avg_time_ms:.0f} predictions/second")


if __name__ == '__main__':
    print("="*80)
    print("ONNX Model Inference Demo")
    print("="*80)
    
    # Example 1: Single prediction with best model
    print("\n" + "="*80)
    print("Example 1: Single Prediction with Best Model")
    print("="*80)
    
    try:
        # Find the best single model (assuming seed 2 from the training output)
        best_model_files = [f for f in os.listdir('.') if f.startswith('best_single_model_')]
        if best_model_files:
            best_model_path = best_model_files[0]
            print(f"\nLoading model: {best_model_path}")
            
            best_session = load_onnx_model(best_model_path)
            
            # Test with a single input
            input_size = 1024
            prediction = predict_single(best_session, input_size)
            
            print(f"\nInput size: {input_size}")
            print(f"Predicted runtime: {prediction:.4f} ms")
        else:
            print("\nNo best single model found. Please run ensemble_top4.py first.")
    except Exception as e:
        print(f"\nError: {e}")
    
    # Example 2: Batch prediction with ensemble model
    print("\n" + "="*80)
    print("Example 2: Batch Prediction with Ensemble Model")
    print("="*80)
    
    try:
        # Find the ensemble model
        ensemble_model_files = [f for f in os.listdir('.') if f.startswith('ensemble_') and f.endswith('.onnx')]
        if ensemble_model_files:
            ensemble_model_path = ensemble_model_files[0]
            print(f"\nLoading model: {ensemble_model_path}")
            
            ensemble_session = load_onnx_model(ensemble_model_path)
            
            # Test with multiple inputs
            input_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
            predictions = predict_batch(ensemble_session, input_sizes)
            
            print(f"\n{'Input Size':<15} {'Predicted Runtime (ms)':<25}")
            print("-"*40)
            for size, pred in zip(input_sizes, predictions):
                print(f"{size:<15} {pred:<25.4f}")
        else:
            print("\nNo ensemble model found. Please run ensemble_top4.py first.")
    except Exception as e:
        print(f"\nError: {e}")
    
    # Example 3: Compare models
    print("\n" + "="*80)
    print("Example 3: Compare Best Single vs Ensemble Models")
    print("="*80)
    
    try:
        if best_model_files and ensemble_model_files:
            test_inputs = [100, 500, 1000, 2000, 5000, 10000]
            compare_models(best_model_files[0], ensemble_model_files[0], test_inputs)
        else:
            print("\nModels not found. Please run ensemble_top4.py first.")
    except Exception as e:
        print(f"\nError: {e}")
    
    # Example 4: Benchmark inference speed
    print("\n" + "="*80)
    print("Example 4: Inference Speed Benchmark")
    print("="*80)
    
    try:
        if ensemble_model_files:
            print(f"\nBenchmarking: {ensemble_model_files[0]}")
            benchmark_inference_speed(ensemble_model_files[0], n_runs=1000)
        else:
            print("\nNo model found for benchmarking.")
    except Exception as e:
        print(f"\nError: {e}")
    
    print("\n" + "="*80)
    print("Demo Completed!")
    print("="*80)
    print("\nYou can now use these ONNX models in any ONNX-compatible framework:")
    print("  - Python: onnxruntime")
    print("  - C++: ONNX Runtime C++ API")
    print("  - Java: ONNX Runtime Java API")
    print("  - C#: ML.NET or ONNX Runtime")
    print("  - JavaScript: ONNX.js")
    print("  - Mobile: Core ML (iOS), NNAPI (Android)")
    print("="*80)
