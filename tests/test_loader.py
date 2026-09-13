import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from config.config import DWRFConfigV4, load_config
from data.loader import PVDataLoader

def test_loader_chronological():
    # Create dummy data
    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randn(100)
    })
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        data.to_csv(f.name, index=False)
        f_path = f.name

    config = load_config('config/config.yaml')
    config.dataset.path = f_path
    config.features.columns = ['feature1', 'feature2']
    config.features.target = 'target'
    config.dataset.cv_type = 'chronological'
    config.dataset.train_ratio = 0.7
    config.dataset.val_ratio = 0.15
    config.dataset.test_ratio = 0.15

    loader = PVDataLoader(config)
    loader.load()
    assert loader.X_train is not None
    assert loader.y_train is not None
    assert loader.X_val is not None
    assert loader.X_test is not None
    assert loader.X_train.shape[0] == 70
    assert loader.X_val.shape[0] == 15
    assert loader.X_test.shape[0] == 15
    # Cleanup
    Path(f_path).unlink()

if __name__ == '__main__':
    test_loader_chronological()
    print("Loader test passed.")