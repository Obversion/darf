import numpy as np
import logging
from pathlib import Path
try:
    import onnx
    import onnxruntime
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)

class ONNXExporter:
    """Export DWRF models to ONNX format for ML.NET compatibility."""

    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.export.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, model, X_sample, feature_names):
        """Export model to ONNX."""
        if not ONNX_AVAILABLE:
            logger.warning("ONNX export not available; install onnx, skl2onnx, onnxruntime.")
            return

        # Determine input type
        initial_type = [('float_input', FloatTensorType([None, X_sample.shape[1]]))]
        try:
            onnx_model = convert_sklearn(model, initial_types=initial_type,
                                         target_opset=self.config.export.onnx.get('opset_version', 14))
        except Exception as e:
            logger.error("Error converting to ONNX: %s", e)
            return

        # Save
        model_path = self.output_dir / 'dwrf_model.onnx'
        with open(model_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        logger.info("ONNX model exported to %s", model_path)

        # Optionally test inference
        try:
            sess = onnxruntime.InferenceSession(str(model_path))
            input_name = sess.get_inputs()[0].name
            pred = sess.run(None, {input_name: X_sample.astype(np.float32)})[0]
            logger.info("ONNX inference test passed.")
        except Exception as e:
            logger.warning("ONNX inference test failed: %s", e)