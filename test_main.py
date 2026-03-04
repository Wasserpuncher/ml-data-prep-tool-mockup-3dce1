import unittest
import pandas as pd
import numpy as np
from main import DataPreprocessor

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
        df_temp['numeric_feature_1'].fillna(df_temp['numeric_feature_1'].mean(), inplace=True)
        df_temp['numeric_feature_2'].fillna(df_temp['numeric_feature_2'].mean(), inplace=True)

        df_processed = self.preprocessor.scale_features(df_temp, columns=['numeric_feature_1', 'numeric_feature_2'])

        # Überprüfe, ob der Mittelwert der skalierten Spalten nahe Null ist.
        self.assertAlmostEqual(df_processed['numeric_feature_1'].mean(), 0.0, places=5)
        self.assertAlmostEqual(df_processed['numeric_feature_2'].mean(), 0.0, places=5)

        # Überprüfe, ob die Standardabweichung der skalierten Spalten nahe Eins ist.
        self.assertAlmostEqual(df_processed['numeric_feature_1'].std(), 1.0, places=5)
        self.assertAlmostEqual(df_processed['numeric_feature_2'].std(), 1.0, places=5)

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
        # Original: 6 Spalten. Entfernt: 2. Hinzugefügt: 3 (A) + 3 (B) = 6. Gesamt: 6 - 2 + 6 = 10.
        # Target-Spalte bleibt erhalten.
        self.assertEqual(df_processed.shape[1], 10)

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
        self.assertAlmostEqual(df_processed['numeric_feature_1'].std(), 1.0, places=5)
        self.assertAlmostEqual(df_processed['numeric_feature_2'].std(), 1.0, places=5)

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
        self.assertAlmostEqual(processed_df['num1'].std(), 1.0, places=5)
        self.assertEqual(processed_df.shape[1], 2)


if __name__ == '__main__':
    unittest.main()
