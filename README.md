# 🚀 ML Data Preprocessing Tool Mockup

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Build Status](https://github.com/your-username/ml-data-prep-tool-mockup/workflows/Python%20application/badge.svg)

## 🌟 Overview

Welcome to the **ML Data Preprocessing Tool Mockup**! This project is an enterprise-ready, open-source conceptual tool designed to streamline the crucial data preprocessing phase in machine learning workflows. It provides a robust, extensible, and user-friendly framework for handling common data challenges such as missing values, feature scaling, and categorical encoding.

Developed with best practices in mind, this mockup demonstrates a clean, object-oriented architecture, comprehensive type hinting, and extensive documentation, making it an excellent starting point for building sophisticated data pipelines. It's built to be easily integrated into larger ML systems and serves as a blueprint for high-quality, maintainable Python projects.

## ✨ Features

*   **Data Loading**: Seamlessly load data from CSV files into Pandas DataFrames.
*   **Missing Value Imputation**: Handle `NaN` values using various strategies (mean, median, most frequent).
*   **Feature Scaling**: Standardize numerical features using `StandardScaler`.
*   **Categorical Encoding**: Convert categorical variables into numerical representations using `OneHotEncoder`.
*   **Pipeline Integration**: Combine multiple preprocessing steps into a single, cohesive workflow.
*   **Robust Error Handling**: Graceful error management for file operations and data transformations.
*   **Logging**: Detailed logging for monitoring processing steps and debugging.
*   **Object-Oriented Design**: Clean, modular, and extensible codebase.
*   **Type Hinting & Docstrings**: Enhanced code readability and maintainability.
*   **Unit Testing**: Comprehensive test suite ensuring reliability and correctness.
*   **CI/CD Integration**: GitHub Actions workflow for automated testing.
*   **Bilingual Documentation**: Readme and Architecture documentation available in both English and German.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ml-data-prep-tool-mockup.git
    cd ml-data-prep-tool-mockup
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

Here's a quick example of how to use the `DataPreprocessor` class:

```python
import pandas as pd
from main import DataPreprocessor
import numpy as np

# 1. Create a sample DataFrame (or load from a file)
data = {
    'Feature1': [10, 20, np.nan, 40, 50],
    'Feature2': [1.1, 2.2, 3.3, 4.4, np.nan],
    'CategoryA': ['A', 'B', 'A', 'C', 'B'],
    'CategoryB': ['X', 'Y', 'X', 'Z', 'Y'],
    'Target': [0, 1, 0, 1, 0]
}
df_sample = pd.DataFrame(data)

print("Original DataFrame:")
print(df_sample)

# 2. Initialize the preprocessor with desired strategies
#    (e.g., 'mean' for missing, 'standard' for scaling, 'onehot' for encoding)
preprocessor = DataPreprocessor(
    missing_strategy='mean',
    scaler_strategy='standard',
    encoder_strategy='onehot'
)

# 3. Preprocess the DataFrame
processed_df = preprocessor.preprocess(df_sample.copy())

print("\nProcessed DataFrame:")
print(processed_df)

# You can also load data from a CSV:
# try:
#     df_from_file = preprocessor.load_data('path/to/your_data.csv')
#     processed_file_df = preprocessor.preprocess(df_from_file)
#     print("\nProcessed DataFrame from file:")
#     print(processed_file_df.head())
# except FileNotFoundError:
#     print("Make sure 'path/to/your_data.csv' exists for file loading example.")
```

## 📚 Documentation

*   **Architecture Deep Dive (English)**: [docs/architecture_en.md](docs/architecture_en.md)
*   **Architektur Detail (Deutsch)**: [docs/architecture_de.md](docs/architecture_de.md)

## 🤝 Contributing

We welcome contributions to this project! Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For any questions or suggestions, please open an issue in the GitHub repository.
