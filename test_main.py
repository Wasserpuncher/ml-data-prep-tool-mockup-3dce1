import unittest
import os
import json
import tempfile
import pandas as pd
import numpy as np
from main import DataPreprocessor, load_config, main

class TestDataPreprocessor(unittest.TestCase):
    """
    Testfälle für die DataPreprocessor-Klasse.
    """

    def setUp(self):
        """
        Richtet einen Test-DataFrame für alle Tests ein.
        """
        # Erstelle einen Beispieldatenrahmen mit fehlenden Werten, numerischen und kategorialen Spalten.
        self.sample_data = {
            'numeric_feature_1': [10, 20, np.nan, 40, 50, 60],
            'numeric_feature_2': [1.1, np.nan, 3.3, 4.4, 5.5, 6.6],
            'categorical_feature_A': ['A', 'B', 'A', 'C', 'B', 'A'],
            'categorical_feature_B': ['X', 'Y', 'X', 'Z', 'Y', 'X'],
            'target': [0, 1, 0, 1, 0, 1] # Eine Zielspalte, die nicht verarbeitet werden sollte.
        }
        self.df = pd.DataFrame(self.sample_data)
        # Initialisiere den Preprocessor für jeden Test neu, um Isolation zu gewährleisten.
        self.preprocessor = DataPreprocessor()

    def test_handle_missing_mean_strategy(self):
        """
        Testet die Behandlung fehlender Werte mit der 'mean'-Strategie.
        """
        # Erstelle einen Preprocessor, der die 'mean'-Strategie verwendet.
        mean_preprocessor = DataPreprocessor(missing_strategy='mean')
        # Kopiere den DataFrame, um den Originalzustand zu bewahren.
        df_processed = mean_preprocessor.handle_missing(self.df.copy())

        # Überprüfe, ob keine NaN-Werte in den numerischen Spalten vorhanden sind.
        self.assertFalse(df_processed['numeric_feature_1'].isnull().any(), "Sollte keine NaN in numeric_feature_1 haben")
        self.assertFalse(df_processed['numeric_feature_2'].isnull().any(), "Sollte keine NaN in numeric_feature_2 haben")

        # Berechne den erwarteten Mittelwert für 'numeric_feature_1' (ohne NaN).
        expected_mean_1 = (10 + 20 + 40 + 50 + 60) / 5
        # Überprüfe, ob der fehlende Wert korrekt mit dem Mittelwert imputiert wurde.
        self.assertAlmostEqual(df_processed.loc[2, 'numeric_feature_1'], expected_mean_1, places=5)

        # Berechne den erwarteten Mittelwert für 'numeric_feature_2' (ohne NaN).
        expected_mean_2 = (1.1 + 3.3 + 4.4 + 5.5 + 6.6) / 5
        # Überprüfe, ob der fehlende Wert korrekt mit dem Mittelwert imputiert wurde.
        self.assertAlmostEqual(df_processed.loc[1, 'numeric_feature_2'], expected_mean_2, places=5)

    def test_scale_features_standard_scaler(self):
        """
        Testet die Skalierung numerischer Merkmale mit StandardScaler.
        """
        # Kopiere den DataFrame und fülle zuerst fehlende Werte, da StandardScaler keine NaNs handhaben kann.
        df_temp = self.df.copy()
        df_temp['numeric_feature_1'] = df_temp['numeric_feature_1'].fillna(df_temp['numeric_feature_1'].mean())
        df_temp['numeric_feature_2'] = df_temp['numeric_feature_2'].fillna(df_temp['numeric_feature_2'].mean())

        df_processed = self.preprocessor.scale_features(df_temp, columns=['numeric_feature_1', 'numeric_feature_2'])

        # Überprüfe, ob der Mittelwert der skalierten Spalten nahe Null ist.
        self.assertAlmostEqual(df_processed['numeric_feature_1'].mean(), 0.0, places=5)
        self.assertAlmostEqual(df_processed['numeric_feature_2'].mean(), 0.0, places=5)

        # Überprüfe, ob die Standardabweichung der skalierten Spalten nahe Eins ist.
        # StandardScaler normiert auf die Populations-Standardabweichung (ddof=0) == 1.
        # pandas' .std() nutzt standardmäßig ddof=1 (Stichprobe) und ergäbe sqrt(n/(n-1)),
        # daher muss hier explizit ddof=0 gemessen werden, um die korrekte Eigenschaft zu prüfen.
        self.assertAlmostEqual(df_processed['numeric_feature_1'].std(ddof=0), 1.0, places=5)
        self.assertAlmostEqual(df_processed['numeric_feature_2'].std(ddof=0), 1.0, places=5)

    def test_encode_categorical_one_hot_encoder(self):
        """
        Testet die One-Hot-Kodierung kategorialer Merkmale.
        """
        df_processed = self.preprocessor.encode_categorical(self.df.copy(), columns=['categorical_feature_A', 'categorical_feature_B'])

        # Überprüfe, ob die ursprünglichen kategorialen Spalten entfernt wurden.
        self.assertNotIn('categorical_feature_A', df_processed.columns)
        self.assertNotIn('categorical_feature_B', df_processed.columns)

        # Überprüfe, ob neue One-Hot-kodierte Spalten erstellt wurden.
        self.assertIn('categorical_feature_A_A', df_processed.columns)
        self.assertIn('categorical_feature_A_B', df_processed.columns)
        self.assertIn('categorical_feature_A_C', df_processed.columns)
        self.assertIn('categorical_feature_B_X', df_processed.columns)
        self.assertIn('categorical_feature_B_Y', df_processed.columns)
        self.assertIn('categorical_feature_B_Z', df_processed.columns)

        # Überprüfe die Anzahl der Spalten nach der Kodierung.
        # Original: 5 Spalten (numeric_feature_1, numeric_feature_2, categorical_feature_A,
        # categorical_feature_B, target). Entfernt: 2 kategoriale. Hinzugefügt: 3 (A) + 3 (B) = 6.
        # Gesamt: 5 - 2 + 6 = 9. Die numerischen Spalten und die Target-Spalte bleiben erhalten.
        self.assertEqual(df_processed.shape[1], 9)

        # Überprüfe, ob die Summe der One-Hot-Spalten pro Zeile 1 ist (für jede ursprüngliche Spalte).
        self.assertTrue(all(df_processed[['categorical_feature_A_A', 'categorical_feature_A_B', 'categorical_feature_A_C']].sum(axis=1) == 1))
        self.assertTrue(all(df_processed[['categorical_feature_B_X', 'categorical_feature_B_Y', 'categorical_feature_B_Z']].sum(axis=1) == 1))

    def test_preprocess_pipeline(self):
        """
        Testet die vollständige Vorverarbeitungspipeline.
        """
        df_processed = self.preprocessor.preprocess(self.df.copy())

        # Überprüfe, dass keine NaN-Werte mehr vorhanden sind.
        self.assertFalse(df_processed.isnull().any().any(), "Sollte keine NaN-Werte nach der Vorverarbeitung haben")

        # Überprüfe, dass die numerischen Spalten skaliert wurden (Mittelwert nahe 0, Stddev nahe 1).
        # Beachte, dass die Namen der numerischen Spalten gleich bleiben.
        self.assertAlmostEqual(df_processed['numeric_feature_1'].mean(), 0.0, places=5)
        self.assertAlmostEqual(df_processed['numeric_feature_2'].mean(), 0.0, places=5)
        # StandardScaler normiert auf die Populations-Standardabweichung (ddof=0) == 1.
        self.assertAlmostEqual(df_processed['numeric_feature_1'].std(ddof=0), 1.0, places=5)
        self.assertAlmostEqual(df_processed['numeric_feature_2'].std(ddof=0), 1.0, places=5)

        # Überprüfe, dass die kategorialen Spalten kodiert wurden.
        self.assertNotIn('categorical_feature_A', df_processed.columns)
        self.assertIn('categorical_feature_A_A', df_processed.columns)

        # Überprüfe die endgültige Spaltenanzahl.
        # 2 numerische (skaliert), 1 Zielspalte, 6 One-Hot-Spalten = 9 Spalten.
        self.assertEqual(df_processed.shape[1], 9)

    def test_load_data(self):
        """
        Testet die Datenladefunktion mit einer temporären Datei.
        """
        # Erstelle eine temporäre CSV-Datei.
        temp_csv_path = 'temp_test_data.csv'
        self.df.to_csv(temp_csv_path, index=False)

        try:
            loaded_df = self.preprocessor.load_data(temp_csv_path)
            # Überprüfe, ob der geladene DataFrame mit dem Original übereinstimmt (ohne NaN-Vergleich).
            # Für genauen Vergleich müssen NaNs behandelt werden oder spezielle Vergleichsmethoden verwendet werden.
            self.assertEqual(loaded_df.shape, self.df.shape)
            self.assertTrue(list(loaded_df.columns) == list(self.df.columns))
        finally:
            # Lösche die temporäre Datei.
            import os
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)

    def test_load_data_file_not_found(self):
        """
        Testet den Fehlerfall, wenn die Datei nicht gefunden wird.
        """
        with self.assertRaises(FileNotFoundError):
            self.preprocessor.load_data('non_existent_file.csv')

    def test_load_data_empty_file(self):
        """
        Testet den Fehlerfall, wenn die Datei leer ist.
        """
        temp_empty_csv_path = 'temp_empty_data.csv'
        open(temp_empty_csv_path, 'w').close() # Erstelle eine leere Datei.

        try:
            with self.assertRaises(pd.errors.EmptyDataError):
                self.preprocessor.load_data(temp_empty_csv_path)
        finally:
            import os
            if os.path.exists(temp_empty_csv_path):
                os.remove(temp_empty_csv_path)

    def test_no_numeric_columns(self):
        """
        Testet das Verhalten, wenn keine numerischen Spalten vorhanden sind.
        """
        df_only_categorical = pd.DataFrame({
            'cat1': ['A', 'B', 'C'],
            'cat2': ['X', 'Y', 'Z']
        })
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.preprocess(df_only_categorical.copy())
        # Es sollten nur die kodierten Spalten und keine numerischen Operationen durchgeführt werden.
        self.assertFalse(processed_df.isnull().any().any())
        self.assertIn('cat1_A', processed_df.columns)
        self.assertNotIn('cat1', processed_df.columns)
        self.assertEqual(processed_df.shape[1], 6) # 3 für cat1, 3 für cat2

    def test_no_categorical_columns(self):
        """
        Testet das Verhalten, wenn keine kategorialen Spalten vorhanden sind.
        """
        df_only_numeric = pd.DataFrame({
            'num1': [1, 2, np.nan],
            'num2': [4, 5, 6]
        })
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.preprocess(df_only_numeric.copy())
        # Es sollten nur die numerischen Operationen durchgeführt werden.
        self.assertFalse(processed_df.isnull().any().any())
        self.assertAlmostEqual(processed_df['num1'].mean(), 0.0, places=5)
        # StandardScaler normiert auf die Populations-Standardabweichung (ddof=0) == 1.
        self.assertAlmostEqual(processed_df['num1'].std(ddof=0), 1.0, places=5)
        self.assertEqual(processed_df.shape[1], 2)


class TestConfig(unittest.TestCase):
    """
    Testfälle für das Laden der JSON-Konfiguration und die konfigurationsgetriebene
    Vorverarbeitung.
    Test cases for loading the JSON configuration and running config-driven
    preprocessing.
    """

    def setUp(self):
        # Kleiner temporärer Datensatz mit numerischen (inkl. NaN) und
        # kategorialen Spalten.
        self.df = pd.DataFrame({
            'num1': [10, 20, np.nan, 40, 50, 60],
            'num2': [1.1, np.nan, 3.3, 4.4, 5.5, 6.6],
            'cat': ['A', 'B', 'A', 'C', 'B', 'A'],
            'target': [0, 1, 0, 1, 0, 1],
        })
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, 'pipeline.json')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def _write_config(self, data):
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def test_load_config_returns_dict(self):
        """load_config liest und validiert die JSON-Datei."""
        self._write_config({
            'missing_strategy': 'median',
            'numeric_columns': ['num1'],
            'steps': ['impute', 'scale'],
        })
        config = load_config(self.config_path)
        self.assertEqual(config['missing_strategy'], 'median')
        self.assertEqual(config['numeric_columns'], ['num1'])

    def test_from_config_applies_strategy_and_steps(self):
        """
        from_config baut einen Preprocessor, der genau die konfigurierten
        Spalten/Schritte nutzt. Hier: nur 'num1' imputieren+skalieren, KEINE
        Kodierung -> 'cat' bleibt als Rohspalte erhalten, 'num2'-NaN bleibt.
        """
        self._write_config({
            'missing_strategy': 'median',
            'numeric_columns': ['num1'],
            'categorical_columns': [],
            'steps': ['impute', 'scale'],
        })
        pre = DataPreprocessor.from_config(self.config_path)
        self.assertEqual(pre.missing_strategy, 'median')
        self.assertEqual(pre.numeric_columns, ['num1'])
        self.assertEqual(pre.steps, ['impute', 'scale'])

        result = pre.preprocess(self.df.copy())

        # 'encode' lief nicht -> die kategoriale Rohspalte ist noch da.
        self.assertIn('cat', result.columns)
        self.assertNotIn('cat_A', result.columns)
        # 'num1' wurde imputiert (kein NaN) und skaliert (Mittelwert ~0, std ~1).
        self.assertFalse(result['num1'].isnull().any())
        self.assertAlmostEqual(result['num1'].mean(), 0.0, places=5)
        self.assertAlmostEqual(result['num1'].std(ddof=0), 1.0, places=5)
        # 'num2' war NICHT in numeric_columns -> unangetastet, NaN bleibt bestehen.
        self.assertTrue(result['num2'].isnull().any())

    def test_from_config_median_imputation_value(self):
        """
        Prüft echt, dass die konfigurierte 'median'-Strategie greift: der
        fehlende Wert in 'num1' wird mit dem Median (nicht dem Mittelwert)
        gefüllt. Median von [10,20,40,50,60] = 40; Mittelwert = 36.
        """
        self._write_config({
            'missing_strategy': 'median',
            'numeric_columns': ['num1'],
            'steps': ['impute'],  # nur imputieren, damit der Rohwert prüfbar bleibt
        })
        pre = DataPreprocessor.from_config(self.config_path)
        result = pre.preprocess(self.df.copy())
        # Der fehlende Wert stand an Index 2.
        self.assertAlmostEqual(result.loc[2, 'num1'], 40.0, places=5)

    def test_from_config_encode_only(self):
        """
        Nur der 'encode'-Schritt läuft: die kategoriale Spalte wird One-Hot
        kodiert, numerische Spalten bleiben roh (inkl. NaN).
        """
        self._write_config({
            'categorical_columns': ['cat'],
            'steps': ['encode'],
        })
        pre = DataPreprocessor.from_config(self.config_path)
        result = pre.preprocess(self.df.copy())
        self.assertNotIn('cat', result.columns)
        self.assertIn('cat_A', result.columns)
        self.assertIn('cat_B', result.columns)
        self.assertIn('cat_C', result.columns)
        # Numerik unangetastet -> NaN in num1 bleibt.
        self.assertTrue(result['num1'].isnull().any())

    def test_default_behaviour_unchanged_without_config(self):
        """
        Ein ohne Konfiguration erstellter Preprocessor verhält sich exakt wie
        zuvor: alle Schritte, automatische Spaltenerkennung.
        """
        pre = DataPreprocessor()
        self.assertEqual(pre.steps, ['impute', 'scale', 'encode'])
        self.assertIsNone(pre.numeric_columns)
        self.assertIsNone(pre.categorical_columns)
        result = pre.preprocess(self.df.copy())
        # Auto-Erkennung: cat kodiert, Numerik skaliert, kein NaN.
        self.assertFalse(result.isnull().any().any())
        self.assertIn('cat_A', result.columns)

    def test_main_cli_with_config_and_io(self):
        """
        End-to-End über die CLI: Config + CSV-Eingabe -> CSV-Ausgabe.
        """
        input_csv = os.path.join(self.tmpdir, 'in.csv')
        output_csv = os.path.join(self.tmpdir, 'out.csv')
        self.df.to_csv(input_csv, index=False)
        self._write_config({
            'numeric_columns': ['num1', 'num2'],
            'categorical_columns': ['cat'],
            'steps': ['impute', 'scale', 'encode'],
        })
        exit_code = main(['--config', self.config_path, '--input', input_csv, '--output', output_csv])
        self.assertEqual(exit_code, 0)
        self.assertTrue(os.path.exists(output_csv))
        out = pd.read_csv(output_csv)
        # Kodierte Spalten vorhanden, keine NaN mehr in den skalierten Spalten.
        self.assertIn('cat_A', out.columns)
        self.assertFalse(out['num1'].isnull().any())

    def test_invalid_step_raises(self):
        """Ungültige Schritte in der Config werden abgelehnt."""
        self._write_config({'steps': ['impute', 'bogus']})
        with self.assertRaises(ValueError):
            load_config(self.config_path)

    def test_unknown_key_raises(self):
        """Unbekannte Schlüssel werden abgelehnt (Tippfehler-Schutz)."""
        self._write_config({'missing_strategyy': 'mean'})
        with self.assertRaises(ValueError):
            load_config(self.config_path)

    def test_missing_config_file_raises(self):
        """Ein nicht existierender Pfad führt zu FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_config(os.path.join(self.tmpdir, 'nope.json'))

    def test_invalid_json_raises(self):
        """Ungültiges JSON führt zu einem ValueError."""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write('{ not valid ')
        with self.assertRaises(ValueError):
            load_config(self.config_path)

    def test_bad_column_type_raises(self):
        """numeric_columns muss eine Liste von Strings sein."""
        self._write_config({'numeric_columns': [1, 2, 3]})
        with self.assertRaises(ValueError):
            load_config(self.config_path)


if __name__ == '__main__':
    unittest.main()
