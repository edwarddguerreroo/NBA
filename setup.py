from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nba_prediction",
    version="1.0.0",
    author="NBA_IABET",
    author_email="iabetbusiness@gmail.com",
    description="Sistema avanzado de predicción de estadísticas NBA con modelos de ML/DL",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Librerías básicas de datos y computación
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "python-dateutil>=2.8.2",
        
        # Machine Learning y Deep Learning
        "scikit-learn>=0.24.2",
        "torch>=1.9.0",
        "xgboost>=1.4.0",
        "lightgbm>=3.2.0",
        "catboost>=0.26.0",
        "optuna>=2.10.0",
        "scikit-optimize>=0.9.0",
        "shap>=0.39.0",
        
        # Visualización
        "matplotlib>=3.4.2",
        "seaborn>=0.11.1",
        "plotly>=5.1.0",
        
        # Jupyter y notebooks
        "jupyter>=1.0.0",
        "ipywidgets>=7.6.3",
        "nbformat>=5.1.3",
        "kaleido>=0.2.1",
        
        # Utilidades y herramientas
        "tqdm>=4.61.2",
        "joblib>=1.0.0",
        "requests>=2.25.0",
        
        # Procesamiento de datos adicional
        "scipy>=1.7.0",
    ],
    extras_require={
        "gpu": [
            "torch-geometric>=2.0.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nba-train=pipelines.train:main",
            "nba-predict=pipelines.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.csv", "*.txt"],
    },
    zip_safe=False,
) 