import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List, Optional, Dict, Any
import logging

# Konfiguriere das Logging-System für die Ausgabe von Informationen und Fehlern.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        encoder_strategy: str = 'onehot'
    ):
        """
        Initialisiert den DataPreprocessor mit den angegebenen Strategien.

        Args:
            missing_strategy (str): Strategie für fehlende Werte ('mean', 'median', 'most_frequent').
            scaler_strategy (str): Strategie für numerische Skalierung ('standard', 'minmax').
            encoder_strategy (str): Strategie für kategoriale Kodierung ('onehot').
        
        Raises:
            ValueError: Wenn eine unbekannte Strategie angegeben wird.
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

        # Initialisiere interne Transformer als None, sie werden bei Bedarf gefittet.
        self._fitted_transformers: Dict[str, Any] = {}
        logging.info("DataPreprocessor initialisiert mit Strategien: Missing=%s, Scaler=%s, Encoder=%s",
                     self.missing_strategy, self.scaler_strategy, self.encoder_strategy)

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
            categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()
            cols_to_encode = categorical_cols
        else:
            # Stelle sicher, dass die angegebenen Spalten im DataFrame existieren und kategorial sind.
            cols_to_encode = [col for col in columns if col in df_copy.columns and (pd.api.types.is_object_dtype(df_copy[col]) or pd.api.types.is_categorical_dtype(df_copy[col]))]
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
        Führt eine vollständige Vorverarbeitung des DataFrames durch, einschließlich:
        1. Behandeln fehlender Werte (numerische Spalten).
        2. Skalieren numerischer Merkmale.
        3. Kodieren kategorialer Merkmale.

        Args:
            df (pd.DataFrame): Der Eingabe-DataFrame.

        Returns:
            pd.DataFrame: Der vollständig vorverarbeitete DataFrame.
        """
        logging.info("Starte die vollständige Vorverarbeitung des DataFrames.")
        processed_df = df.copy()

        # Schritt 1: Numerische und kategoriale Spalten identifizieren
        numeric_cols = processed_df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Schritt 2: Fehlende Werte behandeln
        if numeric_cols:
            processed_df = self.handle_missing(processed_df, columns=numeric_cols)
        else:
            logging.warning("Keine numerischen Spalten für die Imputation gefunden.")

        # Schritt 3: Numerische Merkmale skalieren
        if numeric_cols:
            processed_df = self.scale_features(processed_df, columns=numeric_cols)
        else:
            logging.warning("Keine numerischen Spalten für die Skalierung gefunden.")

        # Schritt 4: Kategoriale Merkmale kodieren
        if categorical_cols:
            processed_df = self.encode_categorical(processed_df, columns=categorical_cols)
        else:
            logging.warning("Keine kategorialen Spalten für die Kodierung gefunden.")

        logging.info("Vorverarbeitung abgeschlossen. End-Shape: %s", processed_df.shape)
        return processed_df


if __name__ == "__main__":
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
