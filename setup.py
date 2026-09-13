from setuptools import setup, find_packages

setup(
    name="domain_weighted_rf",
    version="4.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyyaml>=5.4.0",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
    ],
    entry_points={
        "console_scripts": [
            "dwrf=main:main",
        ],
    },
    python_requires=">=3.8",
)