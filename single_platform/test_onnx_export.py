"""
测试 ONNX 导出功能
验证所有模型类型是否能正确导出和加载
"""

import numpy as np
import sys
import os

def test_onnx_libraries():
    """测试必要的库是否已安装"""
    print("="*80)
    print("Testing ONNX Export Dependencies")
    print("="*80)
    
    missing_libs = []
    
    # Test basic ONNX libraries
    try:
        import onnx
        print(f"✓ onnx version: {onnx.__version__}")
    except ImportError:
        print("✗ onnx not found")
        missing_libs.append("onnx")
    
    try:
        import onnxruntime as ort
        print(f"✓ onnxruntime version: {ort.__version__}")
    except ImportError:
        print("✗ onnxruntime not found")
        missing_libs.append("onnxruntime")
    
    # Test sklearn conversion library
    try:
        import skl2onnx
        print(f"✓ skl2onnx version: {skl2onnx.__version__}")
    except ImportError:
        print("✗ skl2onnx not found")
        missing_libs.append("skl2onnx")
    
    # Test XGBoost conversion library
    try:
        import onnxmltools
        print(f"✓ onnxmltools version: {onnxmltools.__version__}")
    except ImportError:
        print("✗ onnxmltools not found")
        missing_libs.append("onnxmltools")
    
    # Test ML libraries
    try:
        import sklearn
        print(f"✓ scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("✗ scikit-learn not found")
        missing_libs.append("scikit-learn")
    
    try:
        import xgboost
        print(f"✓ xgboost version: {xgboost.__version__}")
    except ImportError:
        print("✗ xgboost not found")
        missing_libs.append("xgboost")
    
    print()
    
    if missing_libs:
        print("❌ Missing libraries:")
        for lib in missing_libs:
            print(f"   - {lib}")
        print()
        print("Please install missing libraries:")
        print(f"   pip install {' '.join(missing_libs)}")
        print()
        print("Or run: .\\install_onnx_requirements.ps1")
        return False
    else:
        print("✅ All required libraries are installed!")
        return True


def test_curve_model_export():
    """测试曲线拟合模型导出"""
    print("\n" + "="*80)
    print("Testing Curve Model Export")
    print("="*80)
    
    try:
        from export_to_onnx import create_cubic_onnx, create_linear_onnx, create_fft_onnx
        import onnx
        import onnxruntime as ort
        
        # Test cubic
        print("\n1. Testing cubic model...")
        params = [1.0, 2.0, 3.0, 4.0]
        model = create_cubic_onnx(params, "test_cubic")
        onnx.save(model, "test_cubic.onnx")
        
        session = ort.InferenceSession("test_cubic.onnx")
        test_input = np.array([[5.0]], dtype=np.float32)
        output = session.run(None, {session.get_inputs()[0].name: test_input})[0]
        expected = params[0]*5**3 + params[1]*5**2 + params[2]*5 + params[3]
        
        print(f"   Input: 5.0")
        print(f"   Expected: {expected:.4f}")
        print(f"   ONNX output: {output[0][0]:.4f}")
        print(f"   Difference: {abs(expected - output[0][0]):.8f}")
        
        if abs(expected - output[0][0]) < 1e-5:
            print("   ✓ Cubic model export successful")
        else:
            print("   ✗ Cubic model export failed")
            return False
        
        os.remove("test_cubic.onnx")
        
        # Test linear
        print("\n2. Testing linear model...")
        params = [2.5, 10.0]
        model = create_linear_onnx(params, "test_linear")
        onnx.save(model, "test_linear.onnx")
        
        session = ort.InferenceSession("test_linear.onnx")
        output = session.run(None, {session.get_inputs()[0].name: test_input})[0]
        expected = params[0]*5 + params[1]
        
        print(f"   Input: 5.0")
        print(f"   Expected: {expected:.4f}")
        print(f"   ONNX output: {output[0][0]:.4f}")
        print(f"   Difference: {abs(expected - output[0][0]):.8f}")
        
        if abs(expected - output[0][0]) < 1e-5:
            print("   ✓ Linear model export successful")
        else:
            print("   ✗ Linear model export failed")
            return False
        
        os.remove("test_linear.onnx")
        
        print("\n✅ All curve model exports successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sklearn_model_export():
    """测试 sklearn 模型导出"""
    print("\n" + "="*80)
    print("Testing sklearn Model Export")
    print("="*80)
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from export_to_onnx import export_sklearn_model_to_onnx
        import onnx
        import onnxruntime as ort
        
        print("\n1. Testing RandomForest model...")
        
        # Create and train a simple model
        X_train = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
        y_train = np.array([2, 4, 6, 8, 10], dtype=np.float32)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_scaled, y_train)
        
        # Export to ONNX
        onnx_model = export_sklearn_model_to_onnx(model, scaler, input_dim=1)
        onnx.save(onnx_model, "test_rf.onnx")
        
        # Test inference
        test_input = np.array([[3.0]], dtype=np.float32)
        
        # Python prediction
        python_pred = model.predict(scaler.transform(test_input))[0]
        
        # ONNX prediction
        session = ort.InferenceSession("test_rf.onnx")
        onnx_pred = session.run(None, {session.get_inputs()[0].name: test_input})[0][0][0]
        
        print(f"   Input: 3.0")
        print(f"   Python prediction: {python_pred:.4f}")
        print(f"   ONNX prediction: {onnx_pred:.4f}")
        print(f"   Difference: {abs(python_pred - onnx_pred):.8f}")
        
        if abs(python_pred - onnx_pred) < 1e-4:
            print("   ✓ RandomForest model export successful")
        else:
            print("   ✗ RandomForest model export failed")
            return False
        
        os.remove("test_rf.onnx")
        
        print("\n✅ sklearn model export successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_xgboost_model_export():
    """测试 XGBoost 模型导出"""
    print("\n" + "="*80)
    print("Testing XGBoost Model Export")
    print("="*80)
    
    try:
        from xgboost import XGBRegressor
        from export_to_onnx import export_xgboost_to_onnx
        import onnx
        import onnxruntime as ort
        
        print("\n1. Testing XGBoost model...")
        
        # Create and train a simple model
        X_train = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
        y_train = np.array([2, 4, 6, 8, 10], dtype=np.float32)
        
        model = XGBRegressor(n_estimators=10, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        
        # Export to ONNX
        onnx_model = export_xgboost_to_onnx(model, input_dim=1)
        onnx.save(onnx_model, "test_xgb.onnx")
        
        # Test inference
        test_input = np.array([[3.0]], dtype=np.float32)
        
        # Python prediction
        python_pred = model.predict(test_input)[0]
        
        # ONNX prediction
        session = ort.InferenceSession("test_xgb.onnx")
        onnx_pred = session.run(None, {session.get_inputs()[0].name: test_input})[0][0][0]
        
        print(f"   Input: 3.0")
        print(f"   Python prediction: {python_pred:.4f}")
        print(f"   ONNX prediction: {onnx_pred:.4f}")
        print(f"   Difference: {abs(python_pred - onnx_pred):.8f}")
        
        if abs(python_pred - onnx_pred) < 1e-4:
            print("   ✓ XGBoost model export successful")
        else:
            print("   ✗ XGBoost model export failed")
            return False
        
        os.remove("test_xgb.onnx")
        
        print("\n✅ XGBoost model export successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("ONNX Export Functionality Test Suite")
    print("="*80)
    print()
    
    # Test 1: Check dependencies
    if not test_onnx_libraries():
        print("\n" + "="*80)
        print("❌ Dependency check failed. Please install missing libraries.")
        print("="*80)
        return False
    
    # Test 2: Curve models
    if not test_curve_model_export():
        print("\n" + "="*80)
        print("❌ Curve model export test failed.")
        print("="*80)
        return False
    
    # Test 3: sklearn models
    if not test_sklearn_model_export():
        print("\n" + "="*80)
        print("❌ sklearn model export test failed.")
        print("="*80)
        return False
    
    # Test 4: XGBoost models
    if not test_xgboost_model_export():
        print("\n" + "="*80)
        print("❌ XGBoost model export test failed.")
        print("="*80)
        return False
    
    # All tests passed
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print()
    print("Your system is ready to export all model types to ONNX format.")
    print("You can now run run_ensemble_top4.py with full ONNX export support.")
    print()
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
