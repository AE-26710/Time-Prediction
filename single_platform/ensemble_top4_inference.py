"""
Top-4 Ensemble Model Inference Script
Demonstrates how to use the trained ensemble model parameters for prediction
"""
import numpy as np

# Define global fitting functions
def cubic_func(N, a, b, c, d):
    return a * N**3 + b * N**2 + c * N + d

def fft_complexity(N, a, b):
    return a * N * np.log2(N) + b

def linear_func(N, a, b):
    return a * N + b

def quad_func(N, a, b, c):
    return a * N**2 + b * N + c


# ========== Model Parameters (Replace with your trained model parameters) ==========
# After training with ensemble_top4.py, copy the parameters from Top-4 models here
# Example structure for MPC on M7 using curve method:

# Top-4 model parameters (seed, function, params)
# You need to manually copy these from the training output
TOP4_MODELS = [
    # (seed, function_name, parameters_tuple)
    # Example:
    # (2, 'cubic', (0.0001, 0.002, 1.5, 10.0)),
    # (1, 'cubic', (0.00012, 0.0018, 1.48, 9.5)),
    # (2025, 'cubic', (0.00011, 0.0019, 1.52, 10.2)),
    # (33550336, 'cubic', (0.00009, 0.0021, 1.45, 9.8)),
]

# Function mapping
FUNCTION_MAP = {
    'cubic': cubic_func,
    'fft': fft_complexity,
    'linear': linear_func,
    'quad': quad_func,
}


def predict_ensemble(input_values, top4_models):
    """
    Use Top-4 ensemble model for prediction
    
    Args:
        input_values: Input value(s) (scalar or array)
        top4_models: List of (seed, function_name, params) tuples
    
    Returns:
        Prediction value(s) (scalar or array)
    """
    if isinstance(input_values, (int, float)):
        input_values = np.array([input_values])
    elif isinstance(input_values, list):
        input_values = np.array(input_values)
    
    predictions = []
    
    for seed, func_name, params in top4_models:
        func = FUNCTION_MAP[func_name]
        y_pred = func(input_values, *params)
        predictions.append(y_pred)
    
    # Return average of Top-4 predictions
    ensemble_prediction = np.mean(predictions, axis=0)
    
    return ensemble_prediction


# ========== Usage Example ==========
if __name__ == '__main__':
    print("="*60)
    print("Top-4 Ensemble Model Inference Example")
    print("="*60)
    
    if not TOP4_MODELS:
        print("\nWARNING: TOP4_MODELS is empty!")
        print("Please run ensemble_top4.py first to train the model,")
        print("then copy the trained parameters to this script.")
        print("\nShowing example usage with dummy parameters...")
        
        # Example with dummy parameters for demonstration
        TOP4_MODELS_DEMO = [
            (2, 'cubic', (0.0001, 0.002, 1.5, 10.0)),
            (1, 'cubic', (0.00012, 0.0018, 1.48, 9.5)),
            (2025, 'cubic', (0.00011, 0.0019, 1.52, 10.2)),
            (33550336, 'cubic', (0.00009, 0.0021, 1.45, 9.8)),
        ]
        
        # Single prediction
        test_input = 100
        prediction = predict_ensemble(test_input, TOP4_MODELS_DEMO)
        print(f"\nSingle prediction:")
        print(f"  Input: {test_input}")
        print(f"  Predicted runtime: {prediction[0]:.4f} ms")
        
        # Batch prediction
        test_inputs = [50, 100, 200, 500, 1000]
        predictions = predict_ensemble(test_inputs, TOP4_MODELS_DEMO)
        print(f"\nBatch prediction:")
        for inp, pred in zip(test_inputs, predictions):
            print(f"  Input: {inp:6d} -> Predicted: {pred:10.4f} ms")
    else:
        # Use actual trained model
        print(f"\nLoaded {len(TOP4_MODELS)} models from Top-4 ensemble")
        print("\nModel seeds:")
        for idx, (seed, func_name, params) in enumerate(TOP4_MODELS, 1):
            print(f"  {idx}. Seed {seed} (function: {func_name})")
        
        # Single prediction
        test_input = 100
        prediction = predict_ensemble(test_input, TOP4_MODELS)
        print(f"\nSingle prediction:")
        print(f"  Input: {test_input}")
        print(f"  Predicted runtime: {prediction[0]:.4f} ms")
        
        # Batch prediction
        test_inputs = [50, 100, 200, 500, 1000]
        predictions = predict_ensemble(test_inputs, TOP4_MODELS)
        print(f"\nBatch prediction:")
        for inp, pred in zip(test_inputs, predictions):
            print(f"  Input: {inp:6d} -> Predicted: {pred:10.4f} ms")
    
    print("\n" + "="*60)
    print("Usage Instructions:")
    print("="*60)
    print("1. Train the ensemble model:")
    print("   python ensemble_top4.py")
    print()
    print("2. Copy the Top-4 model parameters from training output")
    print("   and paste them into TOP4_MODELS list in this script")
    print()
    print("3. Use predict_ensemble() function for inference:")
    print("   prediction = predict_ensemble(input_value, TOP4_MODELS)")
    print("="*60)
