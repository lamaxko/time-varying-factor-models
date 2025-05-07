import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----- LSTM Model: I_t → h_t -----
class StateRNN(nn.Module):
    def __init__(self, input_size, num_states=5, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_states)

    def forward(self, I_t):
        lstm_out, _ = self.lstm(I_t)
        h_t = self.linear(lstm_out)
        return h_t

# ----- Extract States -----
def extract_macro_states(df, seq_len=12):
    df = df.dropna(axis=1)
    data = torch.tensor(df.values, dtype=torch.float32)
    sequences = [data[i - seq_len:i] for i in range(seq_len, len(data))]
    I_t = torch.stack(sequences)
    model = StateRNN(input_size=I_t.shape[2])
    model.eval()
    with torch.no_grad():
        h_t = model(I_t)
    return h_t

# ----- Main -----
if __name__ == "__main__":
    from fred import FredApi
    fred = FredApi()

    macro_id = "RPI"  # Choose macro variable here

    # Run LSTM on stationary macro data
    I_t = fred.data_stationary.dropna(axis=1)
    h_t = extract_macro_states(I_t, seq_len=12)

    # Align dates with LSTM output
    dates = I_t.index[12:]

    # Prepare full hₜ matrix (T, 5)
    h_t_np = h_t[:, -1, :].cpu().numpy()
    h_t_df = pd.DataFrame(h_t_np, index=dates, columns=[f"h{i+1}" for i in range(5)])

    # Generate split labels
    n = len(h_t_df)
    train_end = int(0.6 * n)
    valid_end = int(0.8 * n)
    split_labels = np.array(['Train'] * train_end + ['Valid'] * (valid_end - train_end) + ['Test'] * (n - valid_end))
    h_t_df["split"] = split_labels
    split_colors = {'Train': 'blue', 'Valid': 'orange', 'Test': 'green'}

    # Optional: Plot 5 latent states
    fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    for i, col in enumerate([f"h{i+1}" for i in range(5)]):
        for label, color in split_colors.items():
            mask = h_t_df["split"] == label
            axs[i].scatter(h_t_df.index[mask], h_t_df[col][mask], color=color, label=label if i == 0 else "", s=10)
        axs[i].set_title(f"Latent State {col}")
        axs[i].grid(True)

    axs[0].legend(title="Split")
    plt.tight_layout()
    plt.show()
