# Architecture Deep Dive: ML Data Preprocessing Tool Mockup

## 1. Introduction

This document provides a detailed insight into the architectural design of the ML Data Preprocessing Tool Mockup. The project aims to offer a robust, modular, and extensible solution for common data preparation tasks in machine learning. It emphasizes clarity, maintainability, and enterprise readiness through its object-oriented approach and adherence to best practices.

## 2. Core Components

The central component of this project is the `DataPreprocessor` class, located in `main.py`. This class encapsulates all the logic required for data loading and various preprocessing steps.

### 2.1. `DataPreprocessor` Class

*   **Purpose**: To provide a unified interface for data preprocessing, abstracting away the complexities of underlying scikit-learn transformers.
*   **Initialization (`__init__`)**: The constructor allows for configuring the preprocessing strategies (e.g., `missing_strategy`, `scaler_strategy`, `encoder_strategy`). This promotes flexibility and allows users to define their desired behavior upfront.
    *   **Strategy Validation**: Input strategies are validated to ensure only supported options are used, preventing runtime errors.
    *   **Internal Transformers**: Scikit-learn transformers (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) are initialized lazily (or on first use via helper methods `_get_imputer`, `_get_scaler`, `_get_encoder`) and stored in `_fitted_transformers` dictionary. This design allows for potential future extensions where different instances of transformers might be needed or for serialization of the fitted transformers.

### 2.2. Helper Methods for Transformers (`_get_imputer`, `_get_scaler`, `_get_encoder`)

These private methods ensure that transformer instances are created and reused within the `DataPreprocessor` instance. They centralize the creation logic and can be extended to support more advanced transformer types or custom implementations based on the chosen strategy.

### 2.3. Data Loading (`load_data`)

*   **Functionality**: Responsible for loading data, specifically from CSV files using `pandas.read_csv`.
*   **Error Handling**: Includes robust error handling for `FileNotFoundError`, `pd.errors.EmptyDataError`, and general exceptions during file operations. This makes the tool resilient to common data loading issues.
*   **Return Type**: Always returns a `pandas.DataFrame`, maintaining consistency across the data flow.

### 2.4. Preprocessing Steps

Each preprocessing step is implemented as a separate public method, allowing for granular control and reusability. They all operate on a `pandas.DataFrame` and return a modified `DataFrame`.

*   **`handle_missing(df: pd.DataFrame, columns: Optional[List[str]] = None)`**: 
    *   **Logic**: Uses `SimpleImputer` from scikit-learn. The imputation strategy (mean, median, most_frequent) is determined by `missing_strategy` set during initialization.
    *   **Column Selection**: Can operate on specified columns or automatically detect and apply to all numeric columns if `columns` is `None`.
    *   **In-place vs. Copy**: Operates on a copy of the DataFrame to prevent unintended side effects on the original data.

*   **`scale_features(df: pd.DataFrame, columns: Optional[List[str]] = None)`**: 
    *   **Logic**: Employs `StandardScaler` from scikit-learn. The scaling strategy (`standard`) is configured at initialization.
    *   **Column Selection**: Similar to `handle_missing`, it can target specific numeric columns or all numeric columns.
    *   **Prerequisite**: Assumes missing values are already handled, as scikit-learn scalers generally do not handle NaNs.

*   **`encode_categorical(df: pd.DataFrame, columns: Optional[List[str]] = None)`**: 
    *   **Logic**: Utilizes `OneHotEncoder` from scikit-learn. The encoding strategy (`onehot`) is fixed.
    *   **Column Selection**: Targets specified categorical columns (object or category dtypes) or all such columns if `columns` is `None`.
    *   **Output**: Replaces original categorical columns with new one-hot encoded columns. `handle_unknown='ignore'` is set to gracefully handle unseen categories in future data.

### 2.5. Orchestration (`preprocess`)

*   **Purpose**: This method serves as the main pipeline orchestrator, combining `handle_missing`, `scale_features`, and `encode_categorical` into a single, logical flow.
*   **Order of Operations**: The typical order (impute -> scale -> encode) is followed, which is crucial for correct preprocessing.
*   **Dynamic Column Detection**: It intelligently identifies numeric and categorical columns at the beginning and applies relevant transformations only to those columns, making the pipeline robust to varying datasets.

## 3. Data Flow

1.  **Input**: A raw `pandas.DataFrame` (either loaded from a file or provided directly).
2.  **Missing Value Handling**: Numeric columns are identified, and missing values are imputed based on the configured strategy.
3.  **Feature Scaling**: The same numeric columns (now without NaNs) are then scaled using the configured scaler.
4.  **Categorical Encoding**: Categorical columns are identified, transformed using one-hot encoding, and the original columns are replaced by the new encoded features.
5.  **Output**: A fully preprocessed `pandas.DataFrame` suitable for direct input into a machine learning model.

## 4. Design Principles & Best Practices

*   **Object-Oriented Programming (OOP)**: The `DataPreprocessor` class encapsulates related functionalities, promoting modularity, reusability, and easier maintenance.
*   **Type Hinting**: Extensive use of type hints (`List`, `Optional`, `pd.DataFrame`) improves code readability, enables static analysis, and reduces potential type-related bugs.
*   **Docstrings**: Comprehensive docstrings for classes and methods facilitate understanding and usage, crucial for open-source projects.
*   **Error Handling**: Explicit `try-except` blocks ensure that the application handles expected and unexpected errors gracefully, providing informative log messages.
*   **Logging**: `logging` module is used for tracking the execution flow and highlighting important events or warnings, aiding in debugging and monitoring.
*   **Separation of Concerns**: Each method focuses on a single preprocessing task, making the code easier to understand, test, and extend.
*   **Immutability (Partial)**: Operations return new DataFrames (via `.copy()`) rather than modifying them in-place, reducing side effects and making the data flow more predictable.
*   **Testability**: The modular design and clear interfaces make it straightforward to write unit tests for each component, as demonstrated in `test_main.py`.
*   **Extensibility**: The design allows for easy addition of new preprocessing steps or alternative strategies by adding new methods or extending existing ones.

## 5. Future Enhancements

*   **Configuration File Support**: Implement loading preprocessing steps and parameters from an external `config.json` or YAML file. This would allow users to define complex pipelines without modifying code.
*   **More Scaling/Encoding Strategies**: Add support for `MinMaxScaler`, `RobustScaler`, `LabelEncoder`, `TargetEncoder`, etc.
*   **Feature Engineering Modules**: Introduce modules for creating new features (e.g., polynomial features, interaction terms, date/time features).
*   **Data Validation**: Integrate checks for data quality (e.g., outliers, data type consistency) before preprocessing.
*   **Pipeline Serialization**: Ability to save and load the fitted preprocessor (including all internal transformers) using `pickle` or `joblib` for consistent preprocessing of new data.
*   **Custom Transformers**: A mechanism to easily plug in custom scikit-learn-compatible transformers.
*   **Performance Optimization**: For very large datasets, consider Dask or Spark integrations.
*   **CLI Interface**: A command-line interface for common preprocessing tasks.

## 6. Conclusion

The ML Data Preprocessing Tool Mockup provides a solid foundation for building sophisticated and reliable data preparation pipelines. Its well-structured architecture, adherence to best practices, and focus on extensibility make it a valuable asset for any machine learning project aiming for production readiness.
