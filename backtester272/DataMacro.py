import pandas as pd
import os

class DataMacro:

    def __init__(self):
        self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data_macro"))
        self.load_macro_data()
        self.load_sensitivity_matrix()

    def load_macro_data(self, 
                        cpi_file: str = "cpi.csv", 
                        refi_rate_file: str = "refinancing_rate.csv", 
                        yield_10Y_file: str = "ten_year_bond_yield.csv"):
        """
        Load and format macroeconomic data (CPI, refinancing rate, and ten-year bond yield).
        - Ensures the date is set as the index and sorted.
        """
        cpi_path = os.path.join(self.base_path, cpi_file)
        refi_rate_path = os.path.join(self.base_path, refi_rate_file)
        yield_10Y_path = os.path.join(self.base_path, yield_10Y_file)

        self.cpi = pd.read_csv(cpi_path, index_col=0, parse_dates=True).sort_index().astype(float).pct_change()[1:]
        self.refi_rate = pd.read_csv(refi_rate_path, index_col=0, parse_dates=True).sort_index().astype(float).pct_change()[1:]
        self.yield_10Y = pd.read_csv(yield_10Y_path, index_col=0, parse_dates=True).sort_index().astype(float).pct_change()[1:]

    def load_sensitivity_matrix(self, sensitivity_file: str = "sensi_macro.csv"):
        """
        Load the sensitivity matrix for sectors and macro variables.
        - The matrix describes the sensitivity of each sector to macroeconomic variables.
        """
        sensitivity_path = os.path.join(self.base_path, sensitivity_file)
        self.sensitivity_matrix = pd.read_csv(sensitivity_path, index_col=0).astype(float)

    def compute_weighted_view(self, macro_view: pd.Series) -> pd.Series:
        """
        Compute the weighted macro view based on sensitivities.
        - Normalizes the macro views to avoid scale bias across variables.
        - Multiplies the normalized macro views by the sensitivity matrix to compute sector-level impacts.

        :param macro_view: series with next estimated value for each macro variable (% > average for example)
                           Keys should match the columns of the sensitivity matrix.
        :return: pandas Series with tilts for each sector.
        """
        if not set(macro_view.index).issubset(self.sensitivity_matrix.columns):
            raise ValueError("Macro view keys must match sensitivity matrix columns.")
        
        # Normalize macro views to avoid scale bias
        normalized_macro_view = (macro_view - macro_view.mean()) / macro_view.std()
        
        # Compute tilts by multiplying sensitivities with normalized macro views
        tilts = self.sensitivity_matrix.mul(normalized_macro_view, axis=1)
        tilts = tilts.sum(axis=1)
        
        return tilts