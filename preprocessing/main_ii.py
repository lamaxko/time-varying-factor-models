from apis.fred import FredApi
from preprocessing.stock_factors import FactorCalc
from preprocessing.tensors import TensorPreprocessor
from models.lstm import MacroStateLSTM
from models.gan_improved import SDFGan

def run_model_pipeline() -> 'pd.DataFrame':
    """Preprocesses data, trains the GAN, and returns the enriched output DataFrame."""
    # --- Step 1: Extract latent states ---
    macro_stationary = FredApi().data_stationary.copy()

    state_rnn = MacroStateLSTM(num_states=4, hidden_size=32, seq_len=12)
    h_t = state_rnn.extract_from_dataframe(macro_stationary)

    moment_rnn = MacroStateLSTM(num_states=4, hidden_size=32, seq_len=12)
    h_t_g = moment_rnn.extract_from_dataframe(macro_stationary)

    I_ti = FactorCalc().panel.copy()

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
    panel_out = run_model_pipeline()

    panel_out.to_csv("trained_outputs_full.csv", index=False)
    print("[✓] Saved full panel to 'trained_outputs_full.csv'")
    panel_out[["date", "id_stock", "omega", "g"]].to_csv("trained_weights.csv", index=False)
    print("[✓] Saved weights to 'trained_weights.csv'")
