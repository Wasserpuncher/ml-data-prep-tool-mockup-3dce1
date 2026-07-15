import argparse
import json
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List, Optional, Dict, Any
import logging

# Konfiguriere das Logging-System für die Ausgabe von Informationen und Fehlern.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Standardname der JSON-Konfigurationsdatei, die gesucht wird, wenn kein
# expliziter Pfad angegeben wurde.
# Default name of the JSON configuration file searched for when no explicit
# path is provided.
DEFAULT_CONFIG_FILENAME = "config.json"

# Erlaubte Schlüssel in der Konfigurationsdatei. Unbekannte Schlüssel werden
# abgelehnt, damit Tippfehler nicht stillschweigend ignoriert werden.
# Allowed keys in the configuration file. Unknown keys are rejected so that
# typos are not silently ignored.
_ALLOWED_CONFIG_KEYS = {
    "missing_strategy",
    "scaler_strategy",
    "encoder_strategy",
    "numeric_columns",
    "categorical_columns",
    "steps",
}

# Gültige Vorverarbeitungsschritte in ihrer kanonischen Ausführungsreihenfolge.
# Valid preprocessing steps in their canonical execution order.
_VALID_STEPS = ("impute", "scale", "encode")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Lädt eine JSON-Konfigurationsdatei für die Vorverarbeitung und validiert sie.
    Loads and validates a JSON preprocessing configuration file.

    Alle Schlüssel sind optional; fehlt einer, gilt das Standardverhalten:
    All keys are optional; a missing key falls back to the default behaviour:

        - ``missing_strategy`` (str): 'mean' | 'median' | 'most_frequent'.
        - ``scaler_strategy`` (str): 'standard'.
        - ``encoder_strategy`` (str): 'onehot'.
        - ``numeric_columns`` (list[str] | null): Spalten, die imputiert und
          skaliert werden. ``null``/fehlt => automatische Erkennung.
        - ``categorical_columns`` (list[str] | null): Spalten, die kodiert
          werden. ``null``/fehlt => automatische Erkennung.
        - ``steps`` (list[str]): Teilmenge von ['impute', 'scale', 'encode'];
          fehlt => alle drei Schritte werden ausgeführt.

    Args:
        config_path (str): Pfad zur JSON-Konfigurationsdatei.

    Returns:
        Dict[str, Any]: Das validierte Konfigurations-Dictionary.

    Raises:
        FileNotFoundError: Wenn die Datei nicht existiert.
        ValueError: Bei ungültigem JSON, unbekannten Schlüsseln oder
            unzulässigen Werttypen/-werten.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Konfigurationsdatei '{config_path}' wurde nicht gefunden."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Konfigurationsdatei '{config_path}' enthält kein gültiges JSON: {exc}"
            ) from exc

    if not isinstance(raw, dict):
        raise ValueError(
            "Die Konfiguration muss ein JSON-Objekt (Dictionary) auf oberster Ebene sein."
        )

    unknown = set(raw) - _ALLOWED_CONFIG_KEYS
    if unknown:
        raise ValueError(
            "Unbekannte Konfigurationsschlüssel: "
            + ", ".join(sorted(unknown))
            + ". Erlaubt sind: "
            + ", ".join(sorted(_ALLOWED_CONFIG_KEYS))
            + "."
        )

    # String-Strategien validieren.
    # Validate string strategy fields.
    for key in ("missing_strategy", "scaler_strategy", "encoder_strategy"):
        if key in raw and not isinstance(raw[key], str):
            raise ValueError(
                f"Konfigurationswert für '{key}' muss eine Zeichenkette sein."
            )

    # Spaltenlisten validieren (Liste von Strings oder null).
    # Validate column lists (list of strings or null).
    for key in ("numeric_columns", "categorical_columns"):
        if key in raw and raw[key] is not None:
            value = raw[key]
            if not isinstance(value, list) or not all(isinstance(c, str) for c in value):
                raise ValueError(
                    f"'{key}' muss eine Liste von Spaltennamen (Strings) oder null sein."
                )

    # Schritte validieren.
    # Validate steps.
    if "steps" in raw:
        steps = raw["steps"]
        if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
            raise ValueError("'steps' muss eine Liste von Strings sein.")
        invalid = [s for s in steps if s not in _VALID_STEPS]
        if invalid:
            raise ValueError(
                "Ungültige Schritte: "
                + ", ".join(invalid)
                + ". Erlaubt sind: "
                + ", ".join(_VALID_STEPS)
                + "."
            )

    return raw


def _is_categorical_like(series: pd.Series) -> bool:
    """
    Returns True if the given series should be treated as a categorical (text)
    feature for one-hot encoding.

    Robust across pandas versions: pandas 3.0 introduced a dedicated ``str``
    dtype for text columns, so ``is_object_dtype`` alone (True only for the
    legacy ``object`` dtype) no longer detects them. We therefore also accept
    the string dtype and the categorical dtype. Numeric, boolean and datetime
    columns return False.
    """
    dtype = series.dtype
    return (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )

class DataPreprocessor:
    """
    Eine Klasse zum Vorverarbeiten von Daten für Machine-Learning-Modelle.
    Unterstützt das Laden von Daten, das Behandeln fehlender Werte,
    das Skalieren numerischer Merkmale und das Kodieren kategorialer Merkmale.
    
    Attribute:
        missing_strategy (str): Die Strategie zum Imputieren fehlender Werte ('mean', 'median', 'most_frequent').
        scaler_strategy (str): Die Strategie zum Skalieren numerischer Merkmale ('standard', 'minmax').
        encoder_strategy (str): Die Strategie zum Kodieren kategorialer Merkmale ('onehot').
        _imputer (SimpleImputer): Internes Imputer-Objekt.
        _scaler (StandardScaler): Internes Skalierer-Objekt.
        _encoder (OneHotEncoder): Internes Kodierer-Objekt.
        _fitted_transformers (Dict[str, Any]): Speichert die gefitteten Transformer.
    """

    def __init__(
        self,
        missing_strategy: str = 'mean',
        scaler_strategy: str = 'standard',
        encoder_strategy: str = 'onehot',
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        steps: Optional[List[str]] = None
    ):
        """
        Initialisiert den DataPreprocessor mit den angegebenen Strategien.

        Args:
            missing_strategy (str): Strategie für fehlende Werte ('mean', 'median', 'most_frequent').
            scaler_strategy (str): Strategie für numerische Skalierung ('standard', 'minmax').
            encoder_strategy (str): Strategie für kategoriale Kodierung ('onehot').
            numeric_columns (Optional[List[str]]): Explizite Liste numerischer Spalten,
                die imputiert und skaliert werden sollen. Wenn None, werden numerische
                Spalten in :meth:`preprocess` automatisch erkannt (bisheriges Verhalten).
            categorical_columns (Optional[List[str]]): Explizite Liste kategorialer
                Spalten, die kodiert werden sollen. Wenn None, werden kategoriale
                Spalten in :meth:`preprocess` automatisch erkannt.
            steps (Optional[List[str]]): Teilmenge von ['impute', 'scale', 'encode'],
                die festlegt, welche Schritte :meth:`preprocess` ausführt und in
                welcher Reihenfolge. Wenn None, werden alle drei Schritte ausgeführt.

        Raises:
            ValueError: Wenn eine unbekannte Strategie oder ein unbekannter Schritt
                angegeben wird.
        """
        # Überprüfe und setze die Strategien für die Vorverarbeitung.
        if missing_strategy not in ['mean', 'median', 'most_frequent']:
            raise ValueError("Unbekannte missing_strategy. Wähle 'mean', 'median' oder 'most_frequent'.")
        self.missing_strategy = missing_strategy

        if scaler_strategy not in ['standard']:
            raise ValueError("Unbekannte scaler_strategy. Wähle 'standard'.")
        self.scaler_strategy = scaler_strategy

        if encoder_strategy not in ['onehot']:
            raise ValueError("Unbekannte encoder_strategy. Wähle 'onehot'.")
        self.encoder_strategy = encoder_strategy

        # Optionale, konfigurationsgetriebene Spaltenauswahl. None bedeutet
        # automatische Erkennung (das bisherige Standardverhalten).
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns

        # Auszuführende Schritte validieren und festlegen.
        if steps is None:
            self.steps = list(_VALID_STEPS)
        else:
            invalid = [s for s in steps if s not in _VALID_STEPS]
            if invalid:
                raise ValueError(
                    "Unbekannte Schritte: "
                    + ", ".join(invalid)
                    + ". Erlaubt sind: "
                    + ", ".join(_VALID_STEPS)
                    + "."
                )
            self.steps = list(steps)

        # Initialisiere interne Transformer als None, sie werden bei Bedarf gefittet.
        self._fitted_transformers: Dict[str, Any] = {}
        logging.info("DataPreprocessor initialisiert mit Strategien: Missing=%s, Scaler=%s, Encoder=%s; Schritte=%s",
                     self.missing_strategy, self.scaler_strategy, self.encoder_strategy, self.steps)

    @classmethod
    def from_config(cls, config_path: str) -> "DataPreprocessor":
        """
        Erstellt einen DataPreprocessor aus einer JSON-Konfigurationsdatei.
        Creates a DataPreprocessor from a JSON configuration file.

        Die Konfigurationsdatei legt die Strategien, die zu verarbeitenden
        Spalten und die auszuführenden Schritte fest. Siehe :func:`load_config`
        für das erwartete Schema. Nicht angegebene Felder behalten ihr
        Standardverhalten bei.

        Args:
            config_path (str): Pfad zur JSON-Konfigurationsdatei.

        Returns:
            DataPreprocessor: Eine gemäß der Konfiguration initialisierte Instanz.
        """
        config = load_config(config_path)
        return cls(
            missing_strategy=config.get("missing_strategy", "mean"),
            scaler_strategy=config.get("scaler_strategy", "standard"),
            encoder_strategy=config.get("encoder_strategy", "onehot"),
            numeric_columns=config.get("numeric_columns"),
            categorical_columns=config.get("categorical_columns"),
            steps=config.get("steps"),
        )

    def _get_imputer(self) -> SimpleImputer:
        """
        Gibt das entsprechende Imputer-Objekt basierend auf der missing_strategy zurück.
        """
        # Erstellt oder gibt einen SimpleImputer zurück.
        if 'imputer' not in self._fitted_transformers:
            self._fitted_transformers['imputer'] = SimpleImputer(strategy=self.missing_strategy)
        return self._fitted_transformers['imputer']

    def _get_scaler(self) -> StandardScaler:
        """
        Gibt das entsprechende Scaler-Objekt basierend auf der scaler_strategy zurück.
        """
        # Erstellt oder gibt einen StandardScaler zurück.
        if 'scaler' not in self._fitted_transformers:
            self._fitted_transformers['scaler'] = StandardScaler()
        return self._fitted_transformers['scaler']

    def _get_encoder(self) -> OneHotEncoder:
        """
        Gibt das entsprechende Encoder-Objekt basierend auf der encoder_strategy zurück.
        """
        # Erstellt oder gibt einen OneHotEncoder zurück.
        if 'encoder' not in self._fitted_transformers:
            # handle_unknown='ignore' ist wichtig, um Fehler bei unbekannten Kategorien in Testdaten zu vermeiden.
            self._fitted_transformers['encoder'] = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        return self._fitted_transformers['encoder']

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Lädt Daten aus einer CSV-Datei in einen Pandas DataFrame.

        Args:
            file_path (str): Der Pfad zur CSV-Datei.

        Returns:
            pd.DataFrame: Der geladene DataFrame.

        Raises:
            FileNotFoundError: Wenn die Datei nicht gefunden wird.
            pd.errors.EmptyDataError: Wenn die Datei leer ist.
            Exception: Für andere Ladefehler.
        """
        try:
            # Versuche, die CSV-Datei zu laden.
            df = pd.read_csv(file_path)
            logging.info("Daten erfolgreich von '%s' geladen. Shape: %s", file_path, df.shape)
            return df
        except FileNotFoundError:
            # Behandle den Fall, dass die Datei nicht existiert.
            logging.error("Fehler: Datei nicht gefunden unter '%s'.", file_path)
            raise
        except pd.errors.EmptyDataError:
            # Behandle den Fall, dass die Datei leer ist.
            logging.error("Fehler: Die Datei '%s' ist leer.", file_path)
            raise
        except Exception as e:
            # Fange alle anderen möglichen Fehler beim Laden ab.
            logging.error("Fehler beim Laden der Daten aus '%s': %s", file_path, e)
            raise

    def handle_missing(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Behandelt fehlende Werte in den angegebenen Spalten oder allen numerischen Spalten.
        Die Strategie wird bei der Initialisierung der Klasse festgelegt.

        Args:
            df (pd.DataFrame): Der Eingabe-DataFrame.
            columns (Optional[List[str]]): Eine Liste von Spalten, in denen fehlende Werte behandelt werden sollen.
                                          Wenn None, werden alle numerischen Spalten verwendet.

        Returns:
            pd.DataFrame: Der DataFrame mit behandelten fehlenden Werten.
        """
        df_copy = df.copy()
        # Ermittle die Spalten, die behandelt werden sollen.
        if columns is None:
            # Wähle nur numerische Spalten für die Imputation.
            numeric_cols = df_copy.select_dtypes(include=['number']).columns.tolist()
            cols_to_impute = numeric_cols
        else:
            # Stelle sicher, dass die angegebenen Spalten im DataFrame existieren.
            cols_to_impute = [col for col in columns if col in df_copy.columns]
            if len(cols_to_impute) != len(columns):
                logging.warning("Nicht alle angegebenen Spalten für die Imputation gefunden. Nur vorhandene Spalten werden verwendet.")

        if not cols_to_impute:
            logging.warning("Keine Spalten für die Imputation gefunden oder angegeben. DataFrame bleibt unverändert.")
            return df_copy

        # Hole den Imputer.
        imputer = self._get_imputer()

        try:
            # Fitte den Imputer und transformiere die Daten.
            # Der Imputer erwartet ein 2D-Array, daher reshape.
            df_copy[cols_to_impute] = imputer.fit_transform(df_copy[cols_to_impute])
            logging.info("Fehlende Werte in Spalten %s mit Strategie '%s' behandelt.", cols_to_impute, self.missing_strategy)
        except Exception as e:
            logging.error("Fehler beim Behandeln fehlender Werte: %s", e)
            raise

        return df_copy

    def scale_features(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Skaliert numerische Merkmale in den angegebenen Spalten oder allen numerischen Spalten.
        Die Strategie wird bei der Initialisierung der Klasse festgelegt.

        Args:
            df (pd.DataFrame): Der Eingabe-DataFrame.
            columns (Optional[List[str]]): Eine Liste von Spalten, die skaliert werden sollen.
                                          Wenn None, werden alle numerischen Spalten verwendet.

        Returns:
            pd.DataFrame: Der DataFrame mit skalierten Merkmalen.
        """
        df_copy = df.copy()
        # Ermittle die Spalten, die skaliert werden sollen.
        if columns is None:
            # Wähle nur numerische Spalten für die Skalierung.
            numeric_cols = df_copy.select_dtypes(include=['number']).columns.tolist()
            cols_to_scale = numeric_cols
        else:
            # Stelle sicher, dass die angegebenen Spalten im DataFrame existieren und numerisch sind.
            cols_to_scale = [col for col in columns if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col])]
            if len(cols_to_scale) != len(columns):
                logging.warning("Nicht alle angegebenen Spalten für die Skalierung gefunden oder numerisch. Nur vorhandene numerische Spalten werden verwendet.")

        if not cols_to_scale:
            logging.warning("Keine Spalten für die Skalierung gefunden oder angegeben. DataFrame bleibt unverändert.")
            return df_copy

        # Hole den Scaler.
        scaler = self._get_scaler()

        try:
            # Fitte den Scaler und transformiere die Daten.
            df_copy[cols_to_scale] = scaler.fit_transform(df_copy[cols_to_scale])
            logging.info("Spalten %s mit Strategie '%s' skaliert.", cols_to_scale, self.scaler_strategy)
        except Exception as e:
            logging.error("Fehler beim Skalieren von Merkmalen: %s", e)
            raise

        return df_copy

    def encode_categorical(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Kodiert kategoriale Merkmale in den angegebenen Spalten oder allen kategorialen Spalten
        mittels One-Hot-Kodierung.

        Args:
            df (pd.DataFrame): Der Eingabe-DataFrame.
            columns (Optional[List[str]]): Eine Liste von Spalten, die kodiert werden sollen.
                                          Wenn None, werden alle Objekt- oder Kategorie-Spalten verwendet.

        Returns:
            pd.DataFrame: Der DataFrame mit kodierten kategorialen Merkmalen.
        """
        df_copy = df.copy()
        # Ermittle die Spalten, die kodiert werden sollen.
        if columns is None:
            # Wähle nur kategoriale Spalten für die Kodierung.
            categorical_cols = [col for col in df_copy.columns if _is_categorical_like(df_copy[col])]
            cols_to_encode = categorical_cols
        else:
            # Stelle sicher, dass die angegebenen Spalten im DataFrame existieren und kategorial sind.
            cols_to_encode = [col for col in columns if col in df_copy.columns and _is_categorical_like(df_copy[col])]
            if len(cols_to_encode) != len(columns):
                logging.warning("Nicht alle angegebenen Spalten für die Kodierung gefunden oder kategorial. Nur vorhandene kategoriale Spalten werden verwendet.")

        if not cols_to_encode:
            logging.warning("Keine Spalten für die Kodierung gefunden oder angegeben. DataFrame bleibt unverändert.")
            return df_copy

        # Hole den Encoder.
        encoder = self._get_encoder()

        try:
            # Fitte den Encoder und transformiere die Daten.
            encoded_data = encoder.fit_transform(df_copy[cols_to_encode])
            # Erstelle neue Spaltennamen für die kodierten Merkmale.
            encoded_feature_names = encoder.get_feature_names_out(cols_to_encode)
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=df_copy.index)

            # Entferne die ursprünglichen kategorialen Spalten und füge die neuen hinzu.
            df_processed = df_copy.drop(columns=cols_to_encode)
            df_processed = pd.concat([df_processed, encoded_df], axis=1)
            logging.info("Kategoriale Spalten %s mit Strategie '%s' kodiert. Neue Spalten: %s", cols_to_encode, self.encoder_strategy, encoded_feature_names.tolist())
        except Exception as e:
            logging.error("Fehler beim Kodieren kategorialer Merkmale: %s", e)
            raise

        return df_processed

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Führt die Vorverarbeitung des DataFrames durch. Welche Schritte laufen
        (imputieren, skalieren, kodieren) und auf welchen Spalten, richtet sich
        nach der Konfiguration der Instanz:

        1. Behandeln fehlender Werte (Schritt 'impute', numerische Spalten).
        2. Skalieren numerischer Merkmale (Schritt 'scale').
        3. Kodieren kategorialer Merkmale (Schritt 'encode').

        Sind ``numeric_columns`` bzw. ``categorical_columns`` gesetzt, werden
        genau diese verwendet; andernfalls werden die Spalten automatisch anhand
        ihres Datentyps erkannt (Standardverhalten). ``steps`` legt fest, welche
        der drei Schritte in welcher Reihenfolge ausgeführt werden.

        Args:
            df (pd.DataFrame): Der Eingabe-DataFrame.

        Returns:
            pd.DataFrame: Der vorverarbeitete DataFrame.
        """
        logging.info("Starte die Vorverarbeitung des DataFrames. Schritte: %s", self.steps)
        processed_df = df.copy()

        # Schritt 1: Numerische und kategoriale Spalten bestimmen. Explizit
        # konfigurierte Spalten haben Vorrang; andernfalls automatische Erkennung.
        if self.numeric_columns is not None:
            numeric_cols = [col for col in self.numeric_columns if col in processed_df.columns]
            missing_numeric = [col for col in self.numeric_columns if col not in processed_df.columns]
            if missing_numeric:
                logging.warning("Konfigurierte numerische Spalten fehlen im DataFrame und werden übersprungen: %s", missing_numeric)
        else:
            numeric_cols = processed_df.select_dtypes(include=['number']).columns.tolist()

        if self.categorical_columns is not None:
            categorical_cols = [col for col in self.categorical_columns if col in processed_df.columns]
            missing_categorical = [col for col in self.categorical_columns if col not in processed_df.columns]
            if missing_categorical:
                logging.warning("Konfigurierte kategoriale Spalten fehlen im DataFrame und werden übersprungen: %s", missing_categorical)
        else:
            categorical_cols = [col for col in processed_df.columns if _is_categorical_like(processed_df[col])]

        # Schritt 2: Die konfigurierten Schritte in ihrer Reihenfolge ausführen.
        for step in self.steps:
            if step == 'impute':
                if numeric_cols:
                    processed_df = self.handle_missing(processed_df, columns=numeric_cols)
                else:
                    logging.warning("Keine numerischen Spalten für die Imputation gefunden.")
            elif step == 'scale':
                if numeric_cols:
                    processed_df = self.scale_features(processed_df, columns=numeric_cols)
                else:
                    logging.warning("Keine numerischen Spalten für die Skalierung gefunden.")
            elif step == 'encode':
                if categorical_cols:
                    processed_df = self.encode_categorical(processed_df, columns=categorical_cols)
                else:
                    logging.warning("Keine kategorialen Spalten für die Kodierung gefunden.")

        logging.info("Vorverarbeitung abgeschlossen. End-Shape: %s", processed_df.shape)
        return processed_df


def _run_demo() -> None:
    # Beispielhafte Verwendung des DataPreprocessors.
    # Erstelle einen Dummy-DataFrame für Demonstrationszwecke.
    data = {
        'Feature1': [10, 20, None, 40, 50],
        'Feature2': [1.1, 2.2, 3.3, 4.4, None],
        'CategoryA': ['A', 'B', 'A', 'C', 'B'],
        'CategoryB': ['X', 'Y', 'X', 'Z', 'Y'],
        'Target': [0, 1, 0, 1, 0]
    }
    df_sample = pd.DataFrame(data)

    print("\nOriginal DataFrame:")
    print(df_sample)
    print("\nDataFrame Info:")
    df_sample.info()

    # Initialisiere den Preprocessor.
    # Wir können hier verschiedene Strategien wählen.
    preprocessor = DataPreprocessor(
        missing_strategy='mean',
        scaler_strategy='standard',
        encoder_strategy='onehot'
    )

    try:
        # Führe die vollständige Vorverarbeitung durch.
        # In einem realen Szenario würden wir die Daten aus einer Datei laden.
        # Für diesen Mockup verwenden wir den direkt erstellten DataFrame.
        processed_df = preprocessor.preprocess(df_sample)

        print("\nVerarbeiteter DataFrame:")
        print(processed_df)
        print("\nVerarbeiteter DataFrame Info:")
        processed_df.info()

        # Beispiel für das Laden von Daten (würde eine Datei benötigen)
        # try:
        #     loaded_df = preprocessor.load_data('your_data.csv')
        #     print("\nGeladener DataFrame:")
        #     print(loaded_df.head())
        # except FileNotFoundError:
        #     print("Bitte erstellen Sie 'your_data.csv' für den Lade-Test.")

    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

    # Beispiel mit anderen Strategien (falls vorhanden, z.B. 'median' für missing)
    # preprocessor_median = DataPreprocessor(missing_strategy='median')
    # processed_df_median = preprocessor_median.preprocess(df_sample.copy())
    # print("\nVerarbeiteter DataFrame (Missing Median):")
    # print(processed_df_median)


def main(argv: Optional[list] = None) -> int:
    """
    Kommandozeilen-Einstiegspunkt für das Vorverarbeitungswerkzeug.
    Command-line entry point for the preprocessing tool.

    Mit ``--config`` wird eine JSON-Konfigurationsdatei geladen, die Strategien,
    Spalten und Schritte festlegt (siehe :func:`load_config`). Ohne ``--config``
    wird nach einer ``config.json`` im aktuellen Verzeichnis gesucht. Wird eine
    Konfiguration gefunden, kann mit ``--input`` ein CSV-Datensatz verarbeitet
    und optional mit ``--output`` gespeichert werden. Ohne Konfiguration und
    ohne Eingabedatei wird die eingebaute Demo ausgeführt.

    With ``--config`` a JSON configuration file is loaded that defines
    strategies, columns, and steps (see :func:`load_config`). Without
    ``--config`` the tool looks for a ``config.json`` in the current directory.
    When a configuration is found, ``--input`` can process a CSV dataset and
    ``--output`` optionally stores the result. Without configuration and without
    an input file the built-in demo runs.

    Args:
        argv (Optional[list]): Argumentliste (für Tests). Standardmäßig
            ``sys.argv[1:]``.

    Returns:
        int: Exit-Code (0 bei Erfolg, ungleich 0 bei Fehler).
    """
    parser = argparse.ArgumentParser(
        description="ML Data Preprocessing Tool: konfigurierbare Vorverarbeitung von CSV-Daten."
    )
    parser.add_argument(
        "-c", "--config", metavar="PATH", default=None,
        help=(
            "Pfad zu einer JSON-Konfigurationsdatei (Strategien, Spalten, Schritte). "
            f"Ohne Angabe wird '{DEFAULT_CONFIG_FILENAME}' im aktuellen Verzeichnis "
            "verwendet, falls vorhanden."
        ),
    )
    parser.add_argument(
        "-i", "--input", metavar="CSV", default=None,
        help="Pfad zu einer CSV-Eingabedatei, die vorverarbeitet werden soll.",
    )
    parser.add_argument(
        "-o", "--output", metavar="CSV", default=None,
        help="Optionaler Pfad, unter dem das Ergebnis als CSV gespeichert wird.",
    )
    args = parser.parse_args(argv)

    config_path = args.config
    if config_path is None and os.path.isfile(DEFAULT_CONFIG_FILENAME):
        config_path = DEFAULT_CONFIG_FILENAME

    # Ohne Konfiguration und ohne Eingabedatei: eingebaute Demo (bisheriges Verhalten).
    if config_path is None and args.input is None:
        _run_demo()
        return 0

    try:
        if config_path is not None:
            preprocessor = DataPreprocessor.from_config(config_path)
            logging.info("Konfiguration aus '%s' geladen.", config_path)
        else:
            preprocessor = DataPreprocessor()

        if args.input is None:
            print("Keine Eingabedatei angegeben (--input). Es wurde nur die Konfiguration validiert.")
            return 0

        df = preprocessor.load_data(args.input)
        processed_df = preprocessor.preprocess(df)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Fehler: {exc}")
        return 1

    if args.output is not None:
        processed_df.to_csv(args.output, index=False)
        print(f"Vorverarbeitete Daten nach '{args.output}' geschrieben. Shape: {processed_df.shape}")
    else:
        print(processed_df.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
