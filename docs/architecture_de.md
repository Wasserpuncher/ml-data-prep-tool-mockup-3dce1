# Architektur Detail: ML Datenvorverarbeitungs-Tool-Mockup

## 1. Einführung

Dieses Dokument bietet einen detaillierten Einblick in das architektonische Design des ML Datenvorverarbeitungs-Tool-Mockups. Das Projekt zielt darauf ab, eine robuste, modulare und erweiterbare Lösung für gängige Datenaufbereitungsaufgaben im Machine Learning anzubieten. Es legt Wert auf Klarheit, Wartbarkeit und Unternehmensreife durch seinen objektorientierten Ansatz und die Einhaltung bewährter Praktiken.

## 2. Kernkomponenten

Die zentrale Komponente dieses Projekts ist die Klasse `DataPreprocessor`, die sich in `main.py` befindet. Diese Klasse kapselt die gesamte Logik, die für das Laden von Daten und verschiedene Vorverarbeitungsschritte erforderlich ist.

### 2.1. `DataPreprocessor`-Klasse

*   **Zweck**: Eine einheitliche Schnittstelle für die Datenvorverarbeitung bereitzustellen, die die Komplexität der zugrunde liegenden Scikit-learn-Transformer abstrahiert.
*   **Initialisierung (`__init__`)**: Der Konstruktor ermöglicht die Konfiguration der Vorverarbeitungsstrategien (z.B. `missing_strategy`, `scaler_strategy`, `encoder_strategy`). Dies fördert die Flexibilität und ermöglicht es Benutzern, ihr gewünschtes Verhalten im Voraus zu definieren.
    *   **Strategievalidierung**: Eingegebene Strategien werden validiert, um sicherzustellen, dass nur unterstützte Optionen verwendet werden, wodurch Laufzeitfehler verhindert werden.
    *   **Interne Transformer**: Scikit-learn-Transformer (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) werden verzögert (oder bei der ersten Verwendung über Hilfsmethoden `_get_imputer`, `_get_scaler`, `_get_encoder`) initialisiert und im `_fitted_transformers`-Wörterbuch gespeichert. Dieses Design ermöglicht zukünftige Erweiterungen, bei denen möglicherweise verschiedene Instanzen von Transformern benötigt werden oder für die Serialisierung der gefitteten Transformer.

### 2.2. Hilfsmethoden für Transformer (`_get_imputer`, `_get_scaler`, `_get_encoder`)

Diese privaten Methoden stellen sicher, dass Transformer-Instanzen innerhalb der `DataPreprocessor`-Instanz erstellt und wiederverwendet werden. Sie zentralisieren die Erstellungslogik und können erweitert werden, um fortgeschrittenere Transformer-Typen oder benutzerdefinierte Implementierungen basierend auf der gewählten Strategie zu unterstützen.

### 2.3. Datenladen (`load_data`)

*   **Funktionalität**: Verantwortlich für das Laden von Daten, insbesondere aus CSV-Dateien mit `pandas.read_csv`.
*   **Fehlerbehandlung**: Umfasst eine robuste Fehlerbehandlung für `FileNotFoundError`, `pd.errors.EmptyDataError` und allgemeine Ausnahmen bei Dateivorgängen. Dies macht das Tool widerstandsfähig gegenüber gängigen Problemen beim Laden von Daten.
*   **Rückgabetyp**: Gibt immer einen `pandas.DataFrame` zurück, wodurch die Konsistenz des Datenflusses gewahrt bleibt.

### 2.4. Vorverarbeitungsschritte

Jeder Vorverarbeitungsschritt ist als separate öffentliche Methode implementiert, was eine granulare Kontrolle und Wiederverwendbarkeit ermöglicht. Alle operieren auf einem `pandas.DataFrame` und geben einen modifizierten `DataFrame` zurück.

*   **`handle_missing(df: pd.DataFrame, columns: Optional[List[str]] = None)`**: 
    *   **Logik**: Verwendet `SimpleImputer` aus Scikit-learn. Die Imputationsstrategie (Mittelwert, Median, häufigster Wert) wird durch die bei der Initialisierung festgelegte `missing_strategy` bestimmt.
    *   **Spaltenauswahl**: Kann auf angegebenen Spalten arbeiten oder automatisch alle numerischen Spalten erkennen und anwenden, wenn `columns` `None` ist.
    *   **In-place vs. Kopie**: Operiert auf einer Kopie des DataFrames, um unbeabsichtigte Nebenwirkungen auf die Originaldaten zu verhindern.

*   **`scale_features(df: pd.DataFrame, columns: Optional[List[str]] = None)`**: 
    *   **Logik**: Verwendet `StandardScaler` aus Scikit-learn. Die Skalierungsstrategie (`standard`) wird bei der Initialisierung konfiguriert.
    *   **Spaltenauswahl**: Ähnlich wie bei `handle_missing` kann es auf bestimmte numerische Spalten oder alle numerischen Spalten abzielen.
    *   **Voraussetzung**: Geht davon aus, dass fehlende Werte bereits behandelt wurden, da Scikit-learn-Scaler im Allgemeinen keine NaNs verarbeiten.

*   **`encode_categorical(df: pd.DataFrame, columns: Optional[List[str]] = None)`**: 
    *   **Logik**: Verwendet `OneHotEncoder` aus Scikit-learn. Die Kodierungsstrategie (`onehot`) ist festgelegt.
    *   **Spaltenauswahl**: Zielt auf angegebene kategoriale Spalten (Objekt- oder Kategorie-Datentypen) oder alle solchen Spalten ab, wenn `columns` `None` ist.
    *   **Ausgabe**: Ersetzt ursprüngliche kategoriale Spalten durch neue One-Hot-kodierte Spalten. `handle_unknown='ignore'` ist gesetzt, um ungesehene Kategorien in zukünftigen Daten elegant zu handhaben.

### 2.5. Orchestrierung (`preprocess`)

*   **Zweck**: Diese Methode dient als Haupt-Pipeline-Orchestrator und kombiniert `handle_missing`, `scale_features` und `encode_categorical` in einem einzigen, logischen Fluss.
*   **Reihenfolge der Operationen**: Die typische Reihenfolge (imputieren -> skalieren -> kodieren) wird eingehalten, was für die korrekte Vorverarbeitung entscheidend ist.
*   **Dynamische Spaltenerkennung**: Sie identifiziert zu Beginn intelligent numerische und kategoriale Spalten und wendet relevante Transformationen nur auf diese Spalten an, wodurch die Pipeline robust gegenüber variierenden Datensätzen wird.

## 3. Datenfluss

1.  **Eingabe**: Ein roher `pandas.DataFrame` (entweder aus einer Datei geladen oder direkt bereitgestellt).
2.  **Behandlung fehlender Werte**: Numerische Spalten werden identifiziert, und fehlende Werte werden basierend auf der konfigurierten Strategie imputiert.
3.  **Merkmalskalierung**: Dieselben numerischen Spalten (jetzt ohne NaNs) werden dann mit dem konfigurierten Scaler skaliert.
4.  **Kategoriale Kodierung**: Kategoriale Spalten werden identifiziert, mittels One-Hot-Kodierung transformiert, und die ursprünglichen Spalten werden durch die neuen kodierten Merkmale ersetzt.
5.  **Ausgabe**: Ein vollständig vorverarbeiteter `pandas.DataFrame`, der für die direkte Eingabe in ein Machine-Learning-Modell geeignet ist.

## 4. Designprinzipien & Best Practices

*   **Objektorientierte Programmierung (OOP)**: Die Klasse `DataPreprocessor` kapselt verwandte Funktionalitäten und fördert Modularität, Wiederverwendbarkeit und einfachere Wartung.
*   **Typenhinweise**: Die umfassende Verwendung von Typenhinweisen (`List`, `Optional`, `pd.DataFrame`) verbessert die Lesbarkeit des Codes, ermöglicht statische Analysen und reduziert potenzielle typbezogene Fehler.
*   **Docstrings**: Umfassende Docstrings für Klassen und Methoden erleichtern das Verständnis und die Verwendung, was für Open-Source-Projekte entscheidend ist.
*   **Fehlerbehandlung**: Explizite `try-except`-Blöcke stellen sicher, dass die Anwendung erwartete und unerwartete Fehler elegant behandelt und informative Protokollmeldungen bereitstellt.
*   **Protokollierung**: Das `logging`-Modul wird verwendet, um den Ausführungsfluss zu verfolgen und wichtige Ereignisse oder Warnungen hervorzuheben, was bei der Fehlerbehebung und Überwachung hilft.
*   **Trennung der Belange**: Jede Methode konzentriert sich auf eine einzelne Vorverarbeitungsaufgabe, wodurch der Code leichter zu verstehen, zu testen und zu erweitern ist.
*   **Unveränderlichkeit (partiell)**: Operationen geben neue DataFrames zurück (mittels `.copy()`), anstatt sie in-place zu modifizieren, wodurch Nebenwirkungen reduziert und der Datenfluss vorhersehbarer wird.
*   **Testbarkeit**: Das modulare Design und die klaren Schnittstellen erleichtern das Schreiben von Unit-Tests für jede Komponente, wie in `test_main.py` demonstriert.
*   **Erweiterbarkeit**: Das Design ermöglicht die einfache Hinzufügung neuer Vorverarbeitungsschritte oder alternativer Strategien durch Hinzufügen neuer Methoden oder Erweiterung bestehender.

## 5. Zukünftige Erweiterungen

*   **Unterstützung von Konfigurationsdateien**: Implementierung des Ladens von Vorverarbeitungsschritten und Parametern aus einer externen `config.json`- oder YAML-Datei. Dies würde es Benutzern ermöglichen, komplexe Pipelines zu definieren, ohne Code ändern zu müssen.
*   **Weitere Skalierungs-/Kodierungsstrategien**: Unterstützung für `MinMaxScaler`, `RobustScaler`, `LabelEncoder`, `TargetEncoder` usw. hinzufügen.
*   **Feature-Engineering-Module**: Einführung von Modulen zur Erstellung neuer Features (z.B. polynomiale Features, Interaktionsterme, Datums-/Zeit-Features).
*   **Datenvalidierung**: Integration von Prüfungen der Datenqualität (z.B. Ausreißer, Datentypkonsistenz) vor der Vorverarbeitung.
*   **Pipeline-Serialisierung**: Möglichkeit, den gefitteten Preprocessor (einschließlich aller internen Transformer) mit `pickle` oder `joblib` zu speichern und zu laden, um eine konsistente Vorverarbeitung neuer Daten zu gewährleisten.
*   **Benutzerdefinierte Transformer**: Ein Mechanismus zum einfachen Einbinden benutzerdefinierter Scikit-learn-kompatibler Transformer.
*   **Leistungsoptimierung**: Für sehr große Datensätze Dask- oder Spark-Integrationen in Betracht ziehen.
*   **CLI-Schnittstelle**: Eine Befehlszeilenschnittstelle für gängige Vorverarbeitungsaufgaben.

## 6. Fazit

Das ML Datenvorverarbeitungs-Tool-Mockup bietet eine solide Grundlage für den Aufbau anspruchsvoller und zuverlässiger Datenaufbereitungspipelines. Seine gut strukturierte Architektur, die Einhaltung bewährter Praktiken und der Fokus auf Erweiterbarkeit machen es zu einem wertvollen Gut für jedes Machine-Learning-Projekt, das auf Produktionsreife abzielt.
