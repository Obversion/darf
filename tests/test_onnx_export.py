# Unit tests for ONNX export
import numpy as np
from config.config import load_config
from export.onnx_exporter import ONNXExporter
from models.factory import ModelFactory

def test_onnx_export():
    try:
        import onnx
        import onnxruntime
    except ImportError:
        print("ONNX not installed, skipping test.")
        return
    X = np.random.randn(10, 5).astype(np.float32)
    y = np.random.randn(10)
    config = load_config('config/config.yaml')
    config.export.enabled = True
    config.export.output_dir = './test_models'
    model = ModelFactory.create_model(config, 'bagging', feature_weights=None)
    model.fit(X, y)
    exporter = ONNXExporter(config)
    exporter.export(model, X, ['f{}'.format(i) for i in range(5)])
    # Check if file exists
    import os
    assert os.path.exists('./test_models/dwrf_model.onnx')
    # Cleanup
    import shutil
    shutil.rmtree('./test_models')
    print("ONNX export test passed.")

if __name__ == '__main__':
    test_onnx_export()