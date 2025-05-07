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

# ----- Main Execution -----
if __name__ == "__main__":
    from fred import FredApi
    fred = FredApi()

    # Use cleaned, stationary macro data as LSTM input
    I_t = fred.data_stationary.dropna(axis=1)
    h_t = extract_macro_states(I_t, seq_len=12)

    print("h_t shape:", h_t.shape)
    print("First sequence:\n", h_t[0])

    # Adjust index to match LSTM output
    dates = I_t.index[12:]

    # Extract first LSTM state component
    h_t_df = pd.DataFrame(h_t[:, -1, 0].cpu().numpy(), index=dates, columns=["h1"])

    # ----- Select macro variable -----
    macro_id = "INDPRO"
    macro_level = fred.data[macro_id].loc[dates]              # raw (level)
    macro_stationary = fred.data_stationary[macro_id].loc[dates]  # transformed (diffed)

    # ----- Build plot data -----
    plot_df = pd.DataFrame({
        "macro": macro_level,
        "macro_diff": macro_stationary,
        "h1": h_t_df["h1"]
    })

    # Create color splits for train/valid/test
    n = len(plot_df)
    train_end = int(0.6 * n)
    valid_end = int(0.8 * n)
    colors = np.array(['blue'] * train_end + ['orange'] * (valid_end - train_end) + ['green'] * (n - valid_end))
    plot_df["split"] = colors

    # ----- Plot -----
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axs[0].scatter(plot_df.index, plot_df["macro"], c=plot_df["split"], s=10)
    axs[0].set_title("Observed Macroeconomic Variable")
    axs[0].grid(True)

    axs[1].scatter(plot_df.index, plot_df["macro_diff"], c=plot_df["split"], s=10)
    axs[1].set_title("Suggested Transformation of Macroeconomic Variable")
    axs[1].grid(True)

    axs[2].scatter(plot_df.index, plot_df["h1"], c=plot_df["split"], s=10)
    axs[2].set_title("Fitted Macroeconomic State by LSTM (hₜ)")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

