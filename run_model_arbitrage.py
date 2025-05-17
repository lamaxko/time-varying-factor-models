import os
import pandas as pd
from apis.fred import FredApi
from preprocessing.stock_factors import FactorCalc
from preprocessing.tensors import TensorPreprocessor
from models.lstm import MacroStateLSTM
from models.gan_arbitrage_allowed import SDFGanArbitrage


def run_model_pipeline(syear, eyear) -> pd.DataFrame:
    """Preprocesses data, trains the arbitrage-allowed GAN, and returns the enriched output DataFrame."""
    
    # --- Step 1: Extract macro latent states ---
    macro_stationary = FredApi().data_stationary.copy()

    state_rnn = MacroStateLSTM(num_states=4, hidden_size=32, seq_len=12)
    h_t = state_rnn.extract_from_dataframe(macro_stationary)

    # Dummy second LSTM for compatibility but not used
    moment_rnn = MacroStateLSTM(num_states=4, hidden_size=32, seq_len=12)
    h_t_g = h_t.copy()  # Use the same h_t for both since g is not used

    I_ti = FactorCalc(syear=syear, eyear=eyear, lag=1).panel.copy()

    # --- Step 2: Prepare tensors ---
    x_omega, _, y = TensorPreprocessor(h_t, h_t_g, I_ti).preprocess()

    # --- Step 3: Initialize and train GAN ---
    gan = SDFGanArbitrage(input_dim_omega=x_omega.shape[1])
    gan.fit(x_omega, y, n_epochs=100, inner_steps=5)

    # --- Step 4: Extract predictions ---
    omega_final = gan.extract_outputs(x_omega)

    # --- Step 5: Attach predictions to full panel ---
    preprocessor = TensorPreprocessor(h_t, h_t_g, I_ti)
    _, _, _ = preprocessor.preprocess()
    panel_out = preprocessor.merged.copy()
    panel_out["omega"] = omega_final
    panel_out["g"] = None  # g is not used; keep column for consistency if needed

    return panel_out


if __name__ == "__main__":
    save_dir = r"gan_weights_arbitrage_allowed/"
    os.makedirs(save_dir, exist_ok=True)

    syear = 2000
    test_start_year, test_end_year = 2006, 2023

    for eyear in range(test_start_year, test_end_year + 1):
        panel_out = run_model_pipeline(syear, eyear)

        full_path = f"{save_dir}{syear}-{eyear}_trained_outputs_full_arbitrage_allowed.csv"
        weights_path = f"{save_dir}{syear}-{eyear}_trained_weights_arbitrage_allowed.csv"

        panel_out.to_csv(full_path, index=False)
        panel_out[["date", "id_stock", "omega"]].to_csv(weights_path, index=False)
