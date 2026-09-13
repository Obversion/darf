"""Verify installation and configuration."""
import sys
import yaml
import numpy as np
import pandas as pd
import sklearn
import xgboost
import matplotlib
import seaborn

print("All required packages are available.")
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"XGBoost: {xgboost.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
print(f"seaborn: {seaborn.__version__}")