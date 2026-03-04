# Contributing to ML Data Preprocessing Tool Mockup

We welcome contributions from the community to make this project even better! Whether it's bug reports, feature requests, documentation improvements, or code contributions, your help is greatly appreciated.

Please take a moment to review this document to understand the contribution process.

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [your-email@example.com](mailto:your-email@example.com).

## How to Contribute

### 1. Reporting Bugs

*   Before submitting a bug report, please check if the issue has already been reported or fixed.
*   Open a new issue on GitHub and provide a clear and concise description of the bug.
*   Include steps to reproduce the bug, expected behavior, and actual behavior.
*   Mention your operating system, Python version, and any relevant library versions.

### 2. Suggesting Enhancements / Feature Requests

*   Open a new issue on GitHub.
*   Clearly describe the enhancement or feature you would like to see.
*   Explain why this feature would be useful and how it aligns with the project's goals.
*   Provide examples or use cases if possible.

### 3. Code Contributions

1.  **Fork the repository**: Click the "Fork" button at the top right of the repository page.
2.  **Clone your fork**: 
    ```bash
    git clone https://github.com/your-username/ml-data-prep-tool-mockup.git
    cd ml-data-prep-tool-mockup
    ```
3.  **Create a new branch**: Choose a descriptive branch name (e.g., `feature/add-minmax-scaler`, `bugfix/fix-nan-handling`).
    ```bash
    git checkout -b feature/your-feature-name
    ```
4.  **Set up your development environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
5.  **Make your changes**: 
    *   Follow the existing code style. (We use `flake8` for linting).
    *   Add comments (in German as per project standard) where necessary to explain complex logic.
    *   Ensure all variable names are in English.
    *   Add or update docstrings for new or modified functions/classes.
    *   Write unit tests for your changes in `test_main.py` to ensure correctness and prevent regressions.
    *   Run existing tests to ensure nothing is broken:
        ```bash
        python -m unittest discover
        ```
    *   Run the linter:
        ```bash
        flake8 .
        ```
6.  **Commit your changes**: Write clear and concise commit messages.
    ```bash
    git add .
    git commit -m "feat: Add MinMax scaler option"
    ```
7.  **Push to your fork**: 
    ```bash
    git push origin feature/your-feature-name
    ```
8.  **Create a Pull Request (PR)**: 
    *   Go to the original repository on GitHub and you'll see a prompt to create a new pull request.
    *   Provide a clear title and description for your PR.
    *   Explain the changes you've made, why you made them, and how to test them.
    *   Reference any related issues (e.g., `Closes #123`).

### 4. Documentation Contributions

*   Improvements to `README.md`, `README_de.md`, `docs/architecture_en.md`, `docs/architecture_de.md`, or this `CONTRIBUTING.md` are always welcome.
*   Submit documentation changes via a Pull Request, following the general code contribution guidelines.

## Coding Standards

*   **Python Version**: Python 3.8+
*   **Code Style**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines. We use `flake8` for linting.
*   **Type Hinting**: Use type hints extensively for function arguments and return values.
*   **Docstrings**: Use [PEP 257](https://www.python.org/dev/peps/pep-0257/) style docstrings for all public classes, methods, and functions.
*   **Comments**: Inline comments should be in German, explaining complex logic or design choices for beginners.
*   **Variable Names**: All variable and function names must be in English.
*   **Tests**: Every new feature or bug fix should ideally be accompanied by unit tests.

Thank you for contributing to the ML Data Preprocessing Tool Mockup!
