# 🚀 ML Datenvorverarbeitungs-Tool-Mockup

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Build Status](https://github.com/your-username/ml-data-prep-tool-mockup/workflows/Python%20application/badge.svg)

## 🌟 Übersicht

Willkommen beim **ML Datenvorverarbeitungs-Tool-Mockup**! Dieses Projekt ist ein unternehmensfähiges, quelloffenes Konzept-Tool, das darauf ausgelegt ist, die entscheidende Datenvorverarbeitungsphase in Machine-Learning-Workflows zu optimieren. Es bietet ein robustes, erweiterbares und benutzerfreundliches Framework zur Bewältigung gängiger Datenherausforderungen wie fehlende Werte, Merkmalskalierung und kategoriale Kodierung.

Mit den besten Praktiken im Hinterkopf entwickelt, demonstriert dieses Mockup eine saubere, objektorientierte Architektur, umfassende Typenhinweise und eine ausführliche Dokumentation, was es zu einem hervorragenden Ausgangspunkt für den Aufbau komplexer Datenpipelines macht. Es ist darauf ausgelegt, leicht in größere ML-Systeme integriert zu werden und dient als Blaupause für hochwertige, wartbare Python-Projekte.

## ✨ Funktionen

*   **Datenladen**: Nahtloses Laden von Daten aus CSV-Dateien in Pandas DataFrames.
*   **Imputation fehlender Werte**: Behandelt `NaN`-Werte mit verschiedenen Strategien (Mittelwert, Median, häufigster Wert).
*   **Merkmalskalierung**: Standardisiert numerische Merkmale mit dem `StandardScaler`.
*   **Kategoriale Kodierung**: Konvertiert kategoriale Variablen in numerische Darstellungen mit dem `OneHotEncoder`.
*   **Pipeline-Integration**: Kombiniert mehrere Vorverarbeitungsschritte in einem einzigen, kohärenten Workflow.
*   **Robuste Fehlerbehandlung**: Elegante Fehlerverwaltung für Dateivorgänge und Datentransformationen.
*   **Protokollierung**: Detaillierte Protokollierung zur Überwachung der Verarbeitungsschritte und Fehlerbehebung.
*   **Objektorientiertes Design**: Sauberer, modularer und erweiterbarer Code.
*   **Typenhinweise & Docstrings**: Verbesserte Lesbarkeit und Wartbarkeit des Codes.
*   **Unit-Tests**: Umfassende Testsuite zur Gewährleistung von Zuverlässigkeit und Korrektheit.
*   **CI/CD-Integration**: GitHub Actions-Workflow für automatisierte Tests.
*   **Zweisprachige Dokumentation**: Readme- und Architektur-Dokumentation in Englisch und Deutsch verfügbar.

## 🛠️ Installation

1.  **Das Repository klonen:**
    ```bash
    git clone https://github.com/your-username/ml-data-prep-tool-mockup.git
    cd ml-data-prep-tool-mockup
    ```

2.  **Eine virtuelle Umgebung erstellen (empfohlen):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Unter Windows `venv\Scripts\activate` verwenden
    ```

3.  **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Verwendung

Hier ist ein kurzes Beispiel, wie die Klasse `DataPreprocessor` verwendet wird:

```python
import pandas as pd
from main import DataPreprocessor
import numpy as np

# 1. Einen Beispieldatenrahmen erstellen (oder aus einer Datei laden)
data = {
    'Feature1': [10, 20, np.nan, 40, 50],
    'Feature2': [1.1, 2.2, 3.3, 4.4, np.nan],
    'CategoryA': ['A', 'B', 'A', 'C', 'B'],
    'CategoryB': ['X', 'Y', 'X', 'Z', 'Y'],
    'Target': [0, 1, 0, 1, 0]
}
df_sample = pd.DataFrame(data)

print("Originaler DataFrame:")
print(df_sample)

# 2. Den Preprocessor mit den gewünschten Strategien initialisieren
#    (z.B. 'mean' für fehlende Werte, 'standard' für Skalierung, 'onehot' für Kodierung)
preprocessor = DataPreprocessor(
    missing_strategy='mean',
    scaler_strategy='standard',
    encoder_strategy='onehot'
)

# 3. Den DataFrame vorverarbeiten
processed_df = preprocessor.preprocess(df_sample.copy())

print("\nVerarbeiteter DataFrame:")
print(processed_df)

# Sie können Daten auch aus einer CSV-Datei laden:
# try:
#     df_from_file = preprocessor.load_data('pfad/zu/ihrer_daten.csv')
#     processed_file_df = preprocessor.preprocess(df_from_file)
#     print("\nVerarbeiteter DataFrame aus Datei:")
#     print(processed_file_df.head())
# except FileNotFoundError:
#     print("Stellen Sie sicher, dass 'pfad/zu/ihrer_daten.csv' für das Beispiel zum Laden von Dateien existiert.")
```

## 📚 Dokumentation

*   **Architecture Deep Dive (English)**: [docs/architecture_en.md](docs/architecture_en.md)
*   **Architektur Detail (Deutsch)**: [docs/architecture_de.md](docs/architecture_de.md)

## 🤝 Mitwirken

Wir freuen uns über Beiträge zu diesem Projekt! Bitte beachten Sie unsere [CONTRIBUTING.md](CONTRIBUTING.md) für Richtlinien zum Einstieg.

## 📄 Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert – siehe die Datei [LICENSE](LICENSE) für Details.

## 📧 Kontakt

Bei Fragen oder Anregungen öffnen Sie bitte ein Issue im GitHub-Repository.
