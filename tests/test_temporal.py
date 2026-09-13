import numpy as np
from config.config import load_config
from analysis.temporal import TemporalAnalyzer

def test_temporal_time_of_day():
    config = load_config('config/config.yaml')
    config.problem.type = 'regression'
    analyzer = TemporalAnalyzer(config)
    y_true = np.random.randn(100)
    y_pred_rf = y_true + 0.1 * np.random.randn(100)
    y_pred_dwrf = y_true + 0.05 * np.random.randn(100)
    hours = np.random.randint(0, 24, size=100)
    results = analyzer.analyze_time_of_day(y_true, y_pred_rf, y_pred_dwrf, hours)
    assert set(results.keys()) == {'morning', 'noon', 'afternoon'}
    for bucket in results:
        assert 'rf' in results[bucket]
        assert 'dwrf' in results[bucket]
    print("Temporal test passed.")

if __name__ == '__main__':
    test_temporal_time_of_day()