import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_cubic_onnx(params, model_name="cubic_model"):
    """
    Create ONNX model for cubic function: a*x^3 + b*x^2 + c*x + d
    """
    a, b, c, d = params
    
    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, 1])
    
    # Output
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, 1])
    
    # Constants
    const_a = helper.make_node('Constant', inputs=[], outputs=['a'],
                                value=helper.make_tensor('a', TensorProto.FLOAT, [], [a]))
    const_b = helper.make_node('Constant', inputs=[], outputs=['b'],
                                value=helper.make_tensor('b', TensorProto.FLOAT, [], [b]))
    const_c = helper.make_node('Constant', inputs=[], outputs=['c'],
                                value=helper.make_tensor('c', TensorProto.FLOAT, [], [c]))
    const_d = helper.make_node('Constant', inputs=[], outputs=['d'],
                                value=helper.make_tensor('d', TensorProto.FLOAT, [], [d]))
    const_2 = helper.make_node('Constant', inputs=[], outputs=['two'],
                                value=helper.make_tensor('two', TensorProto.FLOAT, [], [2.0]))
    const_3 = helper.make_node('Constant', inputs=[], outputs=['three'],
                                value=helper.make_tensor('three', TensorProto.FLOAT, [], [3.0]))
    
    # Calculate x^2
    pow2 = helper.make_node('Pow', inputs=['input', 'two'], outputs=['x_squared'])
    
    # Calculate x^3
    pow3 = helper.make_node('Pow', inputs=['input', 'three'], outputs=['x_cubed'])
    
    # Calculate a*x^3
    mul_a = helper.make_node('Mul', inputs=['a', 'x_cubed'], outputs=['a_x3'])
    
    # Calculate b*x^2
    mul_b = helper.make_node('Mul', inputs=['b', 'x_squared'], outputs=['b_x2'])
    
    # Calculate c*x
    mul_c = helper.make_node('Mul', inputs=['c', 'input'], outputs=['c_x'])
    
    # Add terms: a*x^3 + b*x^2
    add1 = helper.make_node('Add', inputs=['a_x3', 'b_x2'], outputs=['sum1'])
    
    # Add: sum1 + c*x
    add2 = helper.make_node('Add', inputs=['sum1', 'c_x'], outputs=['sum2'])
    
    # Add: sum2 + d
    add3 = helper.make_node('Add', inputs=['sum2', 'd'], outputs=['output'])
    
    # Create graph
    graph_def = helper.make_graph(
        [const_a, const_b, const_c, const_d, const_2, const_3,
         pow2, pow3, mul_a, mul_b, mul_c, add1, add2, add3],
        model_name,
        [input_tensor],
        [output_tensor]
    )
    
    # Create model
    model_def = helper.make_model(graph_def, producer_name='time_prediction')
    model_def.ir_version = 9
    model_def.opset_import[0].version = 13
    
    # Check model
    onnx.checker.check_model(model_def)
    
    return model_def


def create_quadratic_onnx(params, model_name="quad_model"):
    """
    Create ONNX model for quadratic function: a*x^2 + b*x + c
    """
    a, b, c = params
    
    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, 1])
    
    # Output
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, 1])
    
    # Constants
    const_a = helper.make_node('Constant', inputs=[], outputs=['a'],
                                value=helper.make_tensor('a', TensorProto.FLOAT, [], [a]))
    const_b = helper.make_node('Constant', inputs=[], outputs=['b'],
                                value=helper.make_tensor('b', TensorProto.FLOAT, [], [b]))
    const_c = helper.make_node('Constant', inputs=[], outputs=['c'],
                                value=helper.make_tensor('c', TensorProto.FLOAT, [], [c]))
    const_2 = helper.make_node('Constant', inputs=[], outputs=['two'],
                                value=helper.make_tensor('two', TensorProto.FLOAT, [], [2.0]))
    
    # Calculate x^2
    pow2 = helper.make_node('Pow', inputs=['input', 'two'], outputs=['x_squared'])
    
    # Calculate a*x^2
    mul_a = helper.make_node('Mul', inputs=['a', 'x_squared'], outputs=['a_x2'])
    
    # Calculate b*x
    mul_b = helper.make_node('Mul', inputs=['b', 'input'], outputs=['b_x'])
    
    # Add: a*x^2 + b*x
    add1 = helper.make_node('Add', inputs=['a_x2', 'b_x'], outputs=['sum1'])
    
    # Add: sum1 + c
    add2 = helper.make_node('Add', inputs=['sum1', 'c'], outputs=['output'])
    
    # Create graph
    graph_def = helper.make_graph(
        [const_a, const_b, const_c, const_2, pow2, mul_a, mul_b, add1, add2],
        model_name,
        [input_tensor],
        [output_tensor]
    )
    
    # Create model
    model_def = helper.make_model(graph_def, producer_name='time_prediction')
    model_def.ir_version = 9
    model_def.opset_import[0].version = 13
    
    # Check model
    onnx.checker.check_model(model_def)
    
    return model_def


def create_linear_onnx(params, model_name="linear_model"):
    """
    Create ONNX model for linear function: a*x + b
    """
    a, b = params
    
    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, 1])
    
    # Output
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, 1])
    
    # Constants
    const_a = helper.make_node('Constant', inputs=[], outputs=['a'],
                                value=helper.make_tensor('a', TensorProto.FLOAT, [], [a]))
    const_b = helper.make_node('Constant', inputs=[], outputs=['b'],
                                value=helper.make_tensor('b', TensorProto.FLOAT, [], [b]))
    
    # Calculate a*x
    mul = helper.make_node('Mul', inputs=['a', 'input'], outputs=['a_x'])
    
    # Add: a*x + b
    add = helper.make_node('Add', inputs=['a_x', 'b'], outputs=['output'])
    
    # Create graph
    graph_def = helper.make_graph(
        [const_a, const_b, mul, add],
        model_name,
        [input_tensor],
        [output_tensor]
    )
    
    # Create model
    model_def = helper.make_model(graph_def, producer_name='time_prediction')
    model_def.ir_version = 9
    model_def.opset_import[0].version = 13
    
    # Check model
    onnx.checker.check_model(model_def)
    
    return model_def


def create_fft_onnx(params, model_name="fft_model"):
    """
    Create ONNX model for FFT complexity: a*x*log2(x) + b
    """
    a, b = params
    
    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, 1])
    
    # Output
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, 1])
    
    # Constants
    const_a = helper.make_node('Constant', inputs=[], outputs=['a'],
                                value=helper.make_tensor('a', TensorProto.FLOAT, [], [a]))
    const_b = helper.make_node('Constant', inputs=[], outputs=['b'],
                                value=helper.make_tensor('b', TensorProto.FLOAT, [], [b]))
    const_2 = helper.make_node('Constant', inputs=[], outputs=['two'],
                                value=helper.make_tensor('two', TensorProto.FLOAT, [], [2.0]))
    
    # Calculate log2(x) = log(x) / log(2)
    log_x = helper.make_node('Log', inputs=['input'], outputs=['log_x'])
    log_2 = helper.make_node('Log', inputs=['two'], outputs=['log_2'])
    log2_x = helper.make_node('Div', inputs=['log_x', 'log_2'], outputs=['log2_x'])
    
    # Calculate x * log2(x)
    x_log2_x = helper.make_node('Mul', inputs=['input', 'log2_x'], outputs=['x_log2_x'])
    
    # Calculate a * x * log2(x)
    a_x_log2_x = helper.make_node('Mul', inputs=['a', 'x_log2_x'], outputs=['a_x_log2_x'])
    
    # Add: a*x*log2(x) + b
    add = helper.make_node('Add', inputs=['a_x_log2_x', 'b'], outputs=['output'])
    
    # Create graph
    graph_def = helper.make_graph(
        [const_a, const_b, const_2, log_x, log_2, log2_x, x_log2_x, a_x_log2_x, add],
        model_name,
        [input_tensor],
        [output_tensor]
    )
    
    # Create model
    model_def = helper.make_model(graph_def, producer_name='time_prediction')
    model_def.ir_version = 9
    model_def.opset_import[0].version = 13
    
    # Check model
    onnx.checker.check_model(model_def)
    
    return model_def


def create_ensemble_onnx(models_info, model_name="ensemble_model"):
    """
    Create ONNX ensemble model that averages predictions from multiple models
    
    Args:
        models_info: List of tuples (seed, func_name, params)
        model_name: Name of the ensemble model
    """
    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, 1])
    
    # Output
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, 1])
    
    nodes = []
    model_outputs = []
    
    # Create sub-models
    for idx, (seed, func_name, params) in enumerate(models_info):
        prefix = f'model_{idx}_'
        
        if func_name == 'cubic':
            a, b, c, d = params
            
            # Constants
            nodes.append(helper.make_node('Constant', inputs=[], outputs=[prefix+'a'],
                                         value=helper.make_tensor('a', TensorProto.FLOAT, [], [a])))
            nodes.append(helper.make_node('Constant', inputs=[], outputs=[prefix+'b'],
                                         value=helper.make_tensor('b', TensorProto.FLOAT, [], [b])))
            nodes.append(helper.make_node('Constant', inputs=[], outputs=[prefix+'c'],
                                         value=helper.make_tensor('c', TensorProto.FLOAT, [], [c])))
            nodes.append(helper.make_node('Constant', inputs=[], outputs=[prefix+'d'],
                                         value=helper.make_tensor('d', TensorProto.FLOAT, [], [d])))
            nodes.append(helper.make_node('Constant', inputs=[], outputs=[prefix+'two'],
                                         value=helper.make_tensor('two', TensorProto.FLOAT, [], [2.0])))
            nodes.append(helper.make_node('Constant', inputs=[], outputs=[prefix+'three'],
                                         value=helper.make_tensor('three', TensorProto.FLOAT, [], [3.0])))
            
            # Calculations
            nodes.append(helper.make_node('Pow', inputs=['input', prefix+'two'], outputs=[prefix+'x_squared']))
            nodes.append(helper.make_node('Pow', inputs=['input', prefix+'three'], outputs=[prefix+'x_cubed']))
            nodes.append(helper.make_node('Mul', inputs=[prefix+'a', prefix+'x_cubed'], outputs=[prefix+'a_x3']))
            nodes.append(helper.make_node('Mul', inputs=[prefix+'b', prefix+'x_squared'], outputs=[prefix+'b_x2']))
            nodes.append(helper.make_node('Mul', inputs=[prefix+'c', 'input'], outputs=[prefix+'c_x']))
            nodes.append(helper.make_node('Add', inputs=[prefix+'a_x3', prefix+'b_x2'], outputs=[prefix+'sum1']))
            nodes.append(helper.make_node('Add', inputs=[prefix+'sum1', prefix+'c_x'], outputs=[prefix+'sum2']))
            nodes.append(helper.make_node('Add', inputs=[prefix+'sum2', prefix+'d'], outputs=[prefix+'pred']))
            
        elif func_name == 'quad':
            a, b, c = params
            
            nodes.append(helper.make_node('Constant', inputs=[], outputs=[prefix+'a'],
                                         value=helper.make_tensor('a', TensorProto.FLOAT, [], [a])))
            nodes.append(helper.make_node('Constant', inputs=[], outputs=[prefix+'b'],
                                         value=helper.make_tensor('b', TensorProto.FLOAT, [], [b])))
            nodes.append(helper.make_node('Constant', inputs=[], outputs=[prefix+'c'],
                                         value=helper.make_tensor('c', TensorProto.FLOAT, [], [c])))
            nodes.append(helper.make_node('Constant', inputs=[], outputs=[prefix+'two'],
                                         value=helper.make_tensor('two', TensorProto.FLOAT, [], [2.0])))
            
            nodes.append(helper.make_node('Pow', inputs=['input', prefix+'two'], outputs=[prefix+'x_squared']))
            nodes.append(helper.make_node('Mul', inputs=[prefix+'a', prefix+'x_squared'], outputs=[prefix+'a_x2']))
            nodes.append(helper.make_node('Mul', inputs=[prefix+'b', 'input'], outputs=[prefix+'b_x']))
            nodes.append(helper.make_node('Add', inputs=[prefix+'a_x2', prefix+'b_x'], outputs=[prefix+'sum1']))
            nodes.append(helper.make_node('Add', inputs=[prefix+'sum1', prefix+'c'], outputs=[prefix+'pred']))
            
        elif func_name == 'linear':
            a, b = params
            
            nodes.append(helper.make_node('Constant', inputs=[], outputs=[prefix+'a'],
                                         value=helper.make_tensor('a', TensorProto.FLOAT, [], [a])))
            nodes.append(helper.make_node('Constant', inputs=[], outputs=[prefix+'b'],
                                         value=helper.make_tensor('b', TensorProto.FLOAT, [], [b])))
            
            nodes.append(helper.make_node('Mul', inputs=[prefix+'a', 'input'], outputs=[prefix+'a_x']))
            nodes.append(helper.make_node('Add', inputs=[prefix+'a_x', prefix+'b'], outputs=[prefix+'pred']))
            
        elif func_name == 'fft':
            a, b = params
            
            nodes.append(helper.make_node('Constant', inputs=[], outputs=[prefix+'a'],
                                         value=helper.make_tensor('a', TensorProto.FLOAT, [], [a])))
            nodes.append(helper.make_node('Constant', inputs=[], outputs=[prefix+'b'],
                                         value=helper.make_tensor('b', TensorProto.FLOAT, [], [b])))
            nodes.append(helper.make_node('Constant', inputs=[], outputs=[prefix+'two'],
                                         value=helper.make_tensor('two', TensorProto.FLOAT, [], [2.0])))
            
            nodes.append(helper.make_node('Log', inputs=['input'], outputs=[prefix+'log_x']))
            nodes.append(helper.make_node('Log', inputs=[prefix+'two'], outputs=[prefix+'log_2']))
            nodes.append(helper.make_node('Div', inputs=[prefix+'log_x', prefix+'log_2'], outputs=[prefix+'log2_x']))
            nodes.append(helper.make_node('Mul', inputs=['input', prefix+'log2_x'], outputs=[prefix+'x_log2_x']))
            nodes.append(helper.make_node('Mul', inputs=[prefix+'a', prefix+'x_log2_x'], outputs=[prefix+'a_x_log2_x']))
            nodes.append(helper.make_node('Add', inputs=[prefix+'a_x_log2_x', prefix+'b'], outputs=[prefix+'pred']))
        
        model_outputs.append(prefix+'pred')
    
    # Average predictions
    if len(model_outputs) == 1:
        nodes.append(helper.make_node('Identity', inputs=[model_outputs[0]], outputs=['output']))
    else:
        # Sum all predictions
        current_sum = model_outputs[0]
        for i in range(1, len(model_outputs)):
            sum_output = f'sum_{i}' if i < len(model_outputs) - 1 else 'total_sum'
            nodes.append(helper.make_node('Add', inputs=[current_sum, model_outputs[i]], outputs=[sum_output]))
            current_sum = sum_output
        
        # Divide by number of models
        n_models = len(model_outputs)
        nodes.append(helper.make_node('Constant', inputs=[], outputs=['n_models'],
                                     value=helper.make_tensor('n_models', TensorProto.FLOAT, [], [float(n_models)])))
        nodes.append(helper.make_node('Div', inputs=['total_sum', 'n_models'], outputs=['output']))
    
    # Create graph
    graph_def = helper.make_graph(
        nodes,
        model_name,
        [input_tensor],
        [output_tensor]
    )
    
    # Create model
    model_def = helper.make_model(graph_def, producer_name='time_prediction')
    model_def.ir_version = 9
    model_def.opset_import[0].version = 13
    
    # Check model
    onnx.checker.check_model(model_def)
    
    return model_def


def test_onnx_model(onnx_path, test_inputs):
    """
    Test ONNX model with sample inputs
    """
    session = ort.InferenceSession(onnx_path)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(None, {input_name: test_inputs})
    
    return outputs[0]


if __name__ == '__main__':
    print("="*80)
    print("ONNX Model Export Tool")
    print("="*80)
    
    # Example: Export cubic model
    print("\nExample 1: Creating cubic model (a*x^3 + b*x^2 + c*x + d)")
    cubic_params = [1.0, 2.0, 3.0, 4.0]  # Replace with your actual parameters
    cubic_model = create_cubic_onnx(cubic_params, "cubic_example")
    onnx.save(cubic_model, "cubic_model.onnx")
    print("  Saved: cubic_model.onnx")
    
    # Test
    test_input = np.array([[5.0]], dtype=np.float32)
    expected = cubic_params[0]*5**3 + cubic_params[1]*5**2 + cubic_params[2]*5 + cubic_params[3]
    pred = test_onnx_model("cubic_model.onnx", test_input)
    print(f"  Test: input=5.0, expected={expected:.4f}, predicted={pred[0][0]:.4f}")
    
    # Example: Export ensemble model
    print("\nExample 2: Creating ensemble model (average of multiple models)")
    ensemble_info = [
        (1, 'cubic', (1.0, 2.0, 3.0, 4.0)),
        (2, 'cubic', (1.1, 2.1, 3.1, 4.1)),
    ]
    ensemble_model = create_ensemble_onnx(ensemble_info, "ensemble_example")
    onnx.save(ensemble_model, "ensemble_model.onnx")
    print("  Saved: ensemble_model.onnx")
    
    # Test
    pred_ensemble = test_onnx_model("ensemble_model.onnx", test_input)
    print(f"  Test: input=5.0, predicted={pred_ensemble[0][0]:.4f}")
    
    print("\n" + "="*80)
    print("Usage Instructions:")
    print("="*80)
    print("1. Run ensemble_top4.py to get model parameters")
    print("2. Copy the TOP4_MODELS output from Step 8")
    print("3. Paste into this script and modify the ensemble_info variable")
    print("4. Run this script to generate ONNX models")
    print("="*80)


# ========= 新增：sklearn 和 XGBoost 模型导出函数 ==========

def export_sklearn_model_to_onnx(model, scaler, input_dim=1, model_name="sklearn_model"):
    """
    Export sklearn model (RandomForest, SVR, MLP) to ONNX format
    
    Args:
        model: Trained sklearn model (RandomForestRegressor, SVR, MLPRegressor)
        scaler: StandardScaler used for preprocessing (can be None)
        input_dim: Input dimension (default 1 for single feature)
        model_name: Name of the ONNX model
    
    Returns:
        ONNX model
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        # Create pipeline with scaler and model
        if scaler is not None:
            pipeline = Pipeline([
                ('scaler', scaler),
                ('model', model)
            ])
        else:
            pipeline = model
        
        # Define initial types
        initial_type = [('input', FloatTensorType([None, input_dim]))]
        
        # Convert to ONNX (no options needed for regression models)
        onnx_model = convert_sklearn(
            pipeline,
            initial_types=initial_type,
            target_opset=13
        )
        onnx_model.ir_version = 9

        return onnx_model
    
    except ImportError as e:
        raise ImportError(
            "skl2onnx is required for exporting sklearn models. "
            "Please install: pip install skl2onnx"
        ) from e


def export_xgboost_to_onnx(model, input_dim=1, model_name="xgboost_model"):
    """
    Export XGBoost model to ONNX format
    
    Args:
        model: Trained XGBRegressor model
        input_dim: Input dimension (default 1 for single feature)
        model_name: Name of the ONNX model
    
    Returns:
        ONNX model
    """
    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
        import numpy as np
        
        # XGBoost requires feature names to follow pattern 'f0', 'f1', etc.
        # We need to retrain/update the model with proper feature names
        # Or use a workaround by getting the booster and setting feature names
        
        # Get the booster from XGBRegressor
        booster = model.get_booster()
        
        # Save and reload with corrected feature names
        import json
        import tempfile
        import os
        
        # Create temp file for booster
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
            booster.save_model(temp_path)
        
        # Load and modify feature names
        with open(temp_path, 'r') as f:
            booster_json = json.load(f)
        
        # Update feature names to f0, f1, f2, ...
        if 'learner' in booster_json and 'feature_names' in booster_json['learner']:
            original_names = booster_json['learner']['feature_names']
            booster_json['learner']['feature_names'] = [f'f{i}' for i in range(len(original_names))]
        
        # Save modified booster
        with open(temp_path, 'w') as f:
            json.dump(booster_json, f)
        
        # Load modified booster
        from xgboost import Booster
        modified_booster = Booster()
        modified_booster.load_model(temp_path)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        # Define initial types - use 'float_input' as temporary name
        initial_type = [('float_input', FloatTensorType([None, input_dim]))]
        
        # Convert modified booster to ONNX
        onnx_model = onnxmltools.convert.convert_xgboost(
            modified_booster,
            initial_types=initial_type,
            target_opset=13
        )
        onnx_model.ir_version = 9
        
        # Rename input from 'float_input' to 'input' for consistency
        if onnx_model.graph.input[0].name == 'float_input':
            onnx_model.graph.input[0].name = 'input'
            # Also update any nodes that reference this input
            for node in onnx_model.graph.node:
                node.input[:] = ['input' if inp == 'float_input' else inp for inp in node.input]
        
        return onnx_model
    
    except ImportError as e:
        raise ImportError(
            "onnxmltools and xgboost are required for exporting XGBoost models. "
            "Please install: pip install onnxmltools xgboost"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert XGBoost model to ONNX: {e}"
        ) from e


def export_hybrid_model_to_onnx(base_func, base_params, res_model, 
                                 input_dim=1, app_name='KF', model_name="hybrid_model"):
    """
    Export hybrid model (curve fitting + residual model) to ONNX format
    
    Args:
        base_func: Base function (cubic_func, linear_func, etc.) or sklearn model
        base_params: Parameters for base function (or None if sklearn model)
        res_model: Residual model (RandomForestRegressor)
        input_dim: Input dimension
        app_name: Application name (KF, FFT, MPC, etc.)
        model_name: Name of the ONNX model
    
    Returns:
        ONNX model combining base and residual predictions
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import pandas as pd
        import os
        import tempfile
        
        # Export residual model to ONNX
        initial_type = [('input', FloatTensorType([None, 1]))]
        res_onnx = convert_sklearn(
            res_model,
            initial_types=initial_type,
            target_opset=13
        )
        res_onnx.ir_version = 9
        
        # For curve-based base: create base ONNX, export separately
        if base_params is not None:
            # Determine function type and create base ONNX
            from scipy.optimize import curve_fit
            # Import the global functions
            import sys
            import os
            
            # Try to identify function type from params length
            if len(base_params) == 4:
                base_onnx = create_cubic_onnx(base_params, f"{model_name}_base")
            elif len(base_params) == 3:
                base_onnx = create_quadratic_onnx(base_params, f"{model_name}_base")
            elif len(base_params) == 2:
                # Could be linear or FFT - check app name
                if app_name == 'FFT':
                    base_onnx = create_fft_onnx(base_params, f"{model_name}_base")
                else:
                    base_onnx = create_linear_onnx(base_params, f"{model_name}_base")
            else:
                raise ValueError(f"Unsupported number of parameters: {len(base_params)}")
            
            # Return separate models instead of combined
            # This avoids opset compatibility issues
            print("  Info: Exporting hybrid model as separate base+residual files")
            print("  Note: Final prediction = base_prediction + residual_prediction")
            return {'base': base_onnx, 'residual': res_onnx, 'type': 'separate_hybrid'}
        
        else:
            # sklearn base model - create a pipeline and export together
            print("  Info: Exporting sklearn-based hybrid model as pipeline")
            
            from sklearn.pipeline import Pipeline
            from sklearn.base import BaseEstimator, TransformerMixin
            
            # Create a custom transformer that adds predictions
            class PredictionAdder(BaseEstimator, TransformerMixin):
                def __init__(self, base_model, res_model):
                    self.base_model = base_model
                    self.res_model = res_model
                
                def fit(self, X, y=None):
                    return self
                
                def transform(self, X):
                    return X
                
                def predict(self, X):
                    base_pred = self.base_model.predict(X)
                    res_pred = self.res_model.predict(X)
                    return base_pred + res_pred
            
            # Create combined model
            combined_model = PredictionAdder(base_func, res_model)
            
            # Export to ONNX using custom converter
            # For now, export as two separate models
            print("  Warning: Hybrid models with sklearn base exported as separate base+residual")
            print("  You need to manually add predictions: final = base_pred + residual_pred")
            
            # Export base model
            base_initial_type = [('input', FloatTensorType([None, 1]))]
            base_onnx = convert_sklearn(
                base_func,
                initial_types=base_initial_type,
                target_opset=13
            )
            base_onnx.ir_version = 9
            
            # Return a dict with both models instead of combined
            return {'base': base_onnx, 'residual': res_onnx, 'type': 'separate_hybrid'}
    
    except ImportError as e:
        raise ImportError(
            "skl2onnx is required for exporting hybrid models. "
            "Please install: pip install skl2onnx"
        ) from e


def create_combined_hybrid_onnx(base_onnx_path, res_onnx_path, model_name="hybrid_combined"):
    """
    Combine base and residual ONNX models into a single hybrid model
    
    Args:
        base_onnx_path: Path to base model ONNX file
        res_onnx_path: Path to residual model ONNX file
        model_name: Name for the combined model
    
    Returns:
        Combined ONNX model
    """
    # Load both models
    base_model = onnx.load(base_onnx_path)
    res_model = onnx.load(res_onnx_path)
    
    # Create new graph combining both
    # Input tensor
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, 1])
    
    # Output tensor
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, 1])
    
    nodes = []
    initializers = []
    
    # Add base model nodes (rename outputs)
    for node in base_model.graph.node:
        new_node = helper.make_node(
            node.op_type,
            inputs=[inp if inp == 'input' else f'base_{inp}' for inp in node.input],
            outputs=[f'base_{out}' for out in node.output],
            name=f'base_{node.name}' if node.name else None
        )
        # Copy attributes
        for attr in node.attribute:
            new_node.attribute.append(attr)
        nodes.append(new_node)
    
    # Add base model initializers
    for init in base_model.graph.initializer:
        new_init = helper.make_tensor(
            f'base_{init.name}',
            init.data_type,
            init.dims,
            init.raw_data if init.raw_data else init.float_data,
            raw=bool(init.raw_data)
        )
        initializers.append(new_init)
    
    # Add residual model nodes (rename inputs/outputs)
    for node in res_model.graph.node:
        new_node = helper.make_node(
            node.op_type,
            inputs=[inp if inp == 'input' else f'res_{inp}' for inp in node.input],
            outputs=[f'res_{out}' for out in node.output],
            name=f'res_{node.name}' if node.name else None
        )
        # Copy attributes
        for attr in node.attribute:
            new_node.attribute.append(attr)
        nodes.append(new_node)
    
    # Add residual model initializers
    for init in res_model.graph.initializer:
        new_init = helper.make_tensor(
            f'res_{init.name}',
            init.data_type,
            init.dims,
            init.raw_data if init.raw_data else init.float_data,
            raw=bool(init.raw_data)
        )
        initializers.append(new_init)
    
    # Add node to combine base and residual predictions
    base_output_name = f'base_{base_model.graph.output[0].name}'
    res_output_name = f'res_{res_model.graph.output[0].name}'
    
    add_node = helper.make_node(
        'Add',
        inputs=[base_output_name, res_output_name],
        outputs=['output'],
        name='combine_predictions'
    )
    nodes.append(add_node)
    
    # Create combined graph
    graph_def = helper.make_graph(
        nodes,
        model_name,
        [input_tensor],
        [output_tensor],
        initializer=initializers
    )
    
    # Create model
    model_def = helper.make_model(graph_def, producer_name='time_prediction_hybrid')
    model_def.ir_version = 9
    model_def.opset_import[0].version = 13
    
    # Add ONNX-ML opset for sklearn models (TreeEnsembleRegressor)
    # Use version 1 for compatibility with opset 13
    onnxml_opset = model_def.opset_import.add()
    onnxml_opset.domain = 'ai.onnx.ml'
    onnxml_opset.version = 1
    
    # Check model (skip strict checking for hybrid models)
    try:
        onnx.checker.check_model(model_def)
    except Exception as e:
        print(f"  Warning: Model validation skipped due to: {e}")
    
    return model_def

