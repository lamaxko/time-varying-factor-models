import os
from apis.fred import FredApi
from preprocessing.stock_factors import FactorCalc
from preprocessing.tensors import TensorPreprocessor
from models.lstm import MacroStateLSTM
from models.gan_improved import SDFGan

def run_model_pipeline(syear, eyear) -> 'pd.DataFrame':
    """Preprocesses data, trains the GAN, and returns the enriched output DataFrame."""
    # --- Step 1: Extract latent states ---
    macro_stationary = FredApi().data_stationary.copy()

    state_rnn = MacroStateLSTM(num_states=4, hidden_size=32, seq_len=12)
    h_t = state_rnn.extract_from_dataframe(macro_stationary)

    moment_rnn = MacroStateLSTM(num_states=4, hidden_size=32, seq_len=12)
    h_t_g = moment_rnn.extract_from_dataframe(macro_stationary)

    I_ti = FactorCalc(syear=syear, eyear=eyear, lag=1).panel.copy()

    # --- Step 2: Prepare tensors ---
    x_omega, x_g, y = TensorPreprocessor(h_t, h_t_g, I_ti).preprocess()

    # --- Step 3: Initialize and train GAN ---
    gan = SDFGan(input_dim_omega=x_omega.shape[1], input_dim_g=x_g.shape[1])
    gan.fit(x_omega, x_g, y, n_epochs=100, inner_steps=5)

    # --- Step 4: Extract predictions ---
    omega_final, g_final = gan.extract_outputs(x_omega, x_g)

    # --- Step 5: Attach to full panel and return ---
    preprocessor = TensorPreprocessor(h_t, h_t_g, I_ti)
    _, _, _ = preprocessor.preprocess()
    panel_out = preprocessor.merged.copy()
    panel_out["omega"] = omega_final
    panel_out["g"] = g_final

    return panel_out


if __name__ == "__main__":
    save_dir = r"gan_weights/"
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

    syear = 2000
    test_start_year, test_end_year = 2006, 2023

    for eyear in range(test_start_year, test_end_year + 1):
        panel_out = run_model_pipeline(syear, eyear)

        panel_out.to_csv(fr"{save_dir}{syear}-{eyear}_trained_outputs_full.csv", index=False)
        panel_out[["date", "id_stock", "omega", "g"]].to_csv(
            fr"{save_dir}{syear}-{eyear}_trained_weights.csv", index=False
        )
