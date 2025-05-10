import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class StateRNN(nn.Module):
    def __init__(self, input_size, num_states=4, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_states)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        h_t = self.linear(lstm_out)
        return h_t

def state_rnn(df, seq_len=12, min_coverage=0.7, min_active_features=20, smooth_window=6):
    """
    Hybrid LSTM for macro latent state extraction with diagnostics.
    """
    df = df.copy()

    # Step 1: Drop low-coverage features
    coverage = df.notna().mean()
    retained_cols = coverage[coverage >= min_coverage].index.tolist()
    print(f"\n[INFO] Total input features: {df.shape[1]}")
    print(f"[INFO] Retained features after coverage filter (>{min_coverage*100:.0f}%): {len(retained_cols)}\n")

    df = df[retained_cols]
    all_cols = df.columns
    num_features = len(all_cols)

    df = df.ffill()

    sequences = []
    valid_indices = []

    for i in range(seq_len, len(df)):
        window = df.iloc[i - seq_len:i]
        valid_cols = [col for col in all_cols if not window[col].isnull().any()]

        if len(valid_cols) >= min_active_features:
            tensor = torch.tensor(window[valid_cols].values, dtype=torch.float32)
            padded = torch.zeros((seq_len, num_features), dtype=torch.float32)
            for j, col in enumerate(valid_cols):
                col_idx = all_cols.get_loc(col)
                padded[:, col_idx] = tensor[:, j]
            sequences.append(padded)
            valid_indices.append(df.index[i])

            # Diagnostic output
            print(f"[Window @ {df.index[i].date()}] Active features used: {len(valid_cols)}")

    if not sequences:
        raise ValueError("No valid windows found. Adjust `min_coverage` or `min_active_features`.")

    I_t = torch.stack(sequences)

    model = StateRNN(input_size=num_features, num_states=4, hidden_size=32)
    model.eval()
    with torch.no_grad():
        h_t = model(I_t)

    h_np = h_t[:, -1, :].cpu().numpy()
    h_df = pd.DataFrame(h_np, index=pd.to_datetime(valid_indices), columns=[f"h{i}" for i in range(h_np.shape[1])])

    h_df = h_df.rolling(window=smooth_window, min_periods=1, center=True).mean()

    print(f"\n[INFO] Final output: {h_df.shape[0]} time points, {h_df.shape[1]} latent states\n")
    return h_df

def plot_latent_states(h_t_df, recession_series, save_path=None):
    # h_t_df = h_t_df.diff()
    COLORS = ["#0b3c5d", "#328cc1", "#6b0f1a", "#c94c4c"]
    MARKERS = ['o', 's', '^', 'D']
    plt.style.use("default")

    # Align recession indicator to h_t_df index range
    recession_series = recession_series.loc[h_t_df.index.min():h_t_df.index.max()]
    recession_series = recession_series.reindex(h_t_df.index, method='pad').fillna(0)

    # Identify contiguous recession periods
    recession_mask = recession_series == 1
    recession_periods = []
    in_recession = False
    for i in range(len(recession_mask)):
        if recession_mask.iloc[i] and not in_recession:
            start = recession_mask.index[i]
            in_recession = True
        elif not recession_mask.iloc[i] and in_recession:
            end = recession_mask.index[i]
            recession_periods.append((start, end))
            in_recession = False
    if in_recession:
        recession_periods.append((start, recession_mask.index[-1]))

    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    for i, col in enumerate([f"h{i}" for i in range(4)]):
        ax = axs[i]
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        x = h_t_df.index
        y = h_t_df[col]
        scatter_idx = [i for i, date in enumerate(x) if date.month == 1]

        # Add grey recession bars
        for start, end in recession_periods:
            ax.axvspan(start, end, color='lightgrey', alpha=0.4, zorder=0)

        # Line
        ax.plot(x, y,
                color=color,
                linestyle='--' if i % 2 == 0 else ':',
                linewidth=2,
                alpha=0.85)

        # Markers
        ax.scatter(x[scatter_idx], y.iloc[scatter_idx],
                   marker=marker,
                   s=40,
                   facecolors='white',
                   edgecolors=color,
                   linewidths=1.2,
                   zorder=3)

        # Styling
        ax.set_title(f"Latent State {col}", fontsize=13, fontweight='bold', color=color, pad=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_alpha(0.3)
        ax.spines['bottom'].set_alpha(0.3)
        ax.tick_params(axis='both', labelsize=10, pad=6)

    axs[-1].set_xlabel("Date", fontsize=12, labelpad=6)
    fig.suptitle("LSTM-Inferred Macro Latent States with Recession Periods", fontsize=16, fontweight='bold', color="#0b3c5d", y=0.94)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def correlate_latent_states_with_inputs(h_df, macro_df, top_n=5, plot=True, save_path=None, sort_cols=False):
    """
    Compute and plot correlations between latent states and macro variables.
    Styled to match LSTM latent state plots.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    joined = h_df.join(macro_df, how='inner').dropna()
    latent_cols = h_df.columns
    macro_cols = macro_df.columns

    corr_matrix = joined.corr().loc[latent_cols, macro_cols]

    # Print top correlations to console
    for h in latent_cols:
        print(f"\nTop correlations for {h}:")
        top = corr_matrix.loc[h].abs().sort_values(ascending=False).head(top_n)
        for var in top.index:
            print(f"  {var:<30}  r = {corr_matrix.loc[h, var]:.3f}")

    if plot or save_path:
        if sort_cols:
            sort_order = corr_matrix.loc["h0"].abs().sort_values(ascending=False).index
            corr_matrix = corr_matrix[sort_order]

        num_vars = len(corr_matrix.columns)
        fig_width = min(22, max(16, num_vars * 0.4))
        fig_height = 1.2 * len(corr_matrix.index)

        plt.figure(figsize=(fig_width, fig_height), dpi=300)
        sns.heatmap(
            corr_matrix,
            annot=False,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            cbar_kws={"shrink": 0.75}
        )
        plt.xticks(rotation=90, ha='center', fontsize=10)
        plt.yticks(fontsize=11)
        plt.title("Correlation Between Latent States and Macroeconomic Variables",
                  fontsize=16, fontweight='bold', color="#0b3c5d", pad=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Match spacing of state plots

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[Saved correlation heatmap] {save_path}")
        else:
            plt.show()

    return corr_matrix

if __name__ == "__main__":
    from fred import FredApi

    BASE_DIR = r"C:\Users\lasse.kock\Desktop\ms_thesis\code\lstm_plots_clean"
    OUTPUT_DIR = os.path.join(BASE_DIR, "rw")
    CORR_DIR = os.path.join(BASE_DIR, "rw_corr")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CORR_DIR, exist_ok=True)

    fred = FredApi()
    recession = fred.recession_indicator()

    N = 30 # Number of LSTM runs
    for i in range(1, N + 1):
        print(f"\n[Run {i}]")
        h_t_df = state_rnn(fred.data_stationary)

        # Save latent state plot
        img_path = os.path.join(OUTPUT_DIR, f"latent_states_{i}.png")
        plot_latent_states(h_t_df, recession, save_path=img_path)
        print(f"Saved: {img_path}")

        # Save correlation heatmap
        corr_path = os.path.join(CORR_DIR, f"correlations_{i}.png")
        correlate_latent_states_with_inputs(
            h_t_df,
            fred.data_stationary,
            top_n=6,
            plot=True,
            save_path=corr_path
        )
