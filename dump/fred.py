import requests
import pandas as pd
import numpy as np
from io import StringIO
from statsmodels.tsa.stattools import adfuller
from stationarity import stationary


class FredApi:
    def __init__(self):
        self.url = (
            "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
            "fred-md/monthly/current.csv?sc_lang=en&hash=80445D12401C59CF716410F3F7863B64"
        )
        self.data_raw = self.fetch_data()
        self.tcode_row = self.data_raw.iloc[0]
        self.data = self.clean_data()
        self.tcodes = self.get_tcodes()

        self.data_stationary = self.apply_stationarity()

    def fetch_data(self):
        response = requests.get(self.url)
        response.raise_for_status()
        csv_data = StringIO(response.text)
        return pd.read_csv(csv_data)

    def clean_data(self):
        df = self.data_raw.drop(index=0).reset_index(drop=True)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        return df.set_index(df.columns[0])

    def get_tcodes(self):
        return {
            col: int(float(val))
            for col, val in self.tcode_row.items()
            if col != self.data_raw.columns[0] and pd.notna(val) and str(val).strip().replace('.', '', 1).isdigit()
        }

    def apply_stationarity(self):
        df_transformed = pd.DataFrame(index=self.data.index)

        for col in self.data.columns:
            tcode = self.tcodes.get(col)
            series = self.data[col]
            
            try:
                tcode = int(tcode)
                transformed = stationary(series, tcode=tcode)
                df_transformed[col] = transformed
            except Exception as e:
                print(f"Warning: Could not transform {col} (tcode={tcode}): {e}")
                df_transformed[col] = np.nan  # Fill with NaN if transformation fails

        return df_transformed


    def adf_test(self, alpha=0.05):
        results = []

        for col in self.data_stationary.columns:
            series = self.data_stationary[col].replace([np.inf, -np.inf], np.nan).dropna()

            if len(series) < 20:
                results.append((col, None, 'Too short'))
                continue

            try:
                adf_result = adfuller(series, autolag='AIC')
                p_value = adf_result[1]
                status = 'Stationary' if p_value < alpha else 'Non-stationary'
                results.append((col, p_value, status))
            except Exception as e:
                results.append((col, None, f"Error: {str(e)}"))

        df_results = pd.DataFrame(results, columns=["Variable", "p-value", "Status"])
        return df_results.sort_values("p-value", na_position='last')
            

if __name__ == "__main__":
    fred = FredApi()
    print("FRED-MD Raw Data (head):")
    print(fred.data.head())
    print("\nT-Codes:")
    print(fred.tcodes)
    print("\nADF Test (head):")
    print(fred.adf_test())
    print("\nStationary Data (head):")
    print(fred.data_stationary.head())
