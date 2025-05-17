
# Improved GAN Training Loop from Chen, Pelger, and Zhu (2020)

import torch
from models.lstm import MacroStateLSTM
from models.ffn import FeedForwardNet
from apis.fred import FredApi
from preprocessing.stock_factors import FactorCalc
from preprocessing.tensors import TensorPreprocessor

# Step 1: Data and states
macro_stationary = FredApi().data_stationary.copy()

state_rnn = MacroStateLSTM(num_states=4, hidden_size=32, seq_len=12)
h_t = state_rnn.extract_from_dataframe(macro_stationary)

moment_rnn = MacroStateLSTM(num_states=4, hidden_size=32, seq_len=12)
h_t_g = moment_rnn.extract_from_dataframe(macro_stationary)

I_ti = FactorCalc().panel.copy()

x_omega, x_g, y = TensorPreprocessor(h_t, h_t_g, I_ti).preprocess()

# Step 2: Initialize Networks
sdf_net = FeedForwardNet(input_dim=x_omega.shape[1], hidden_dim=64, output_dim=1)
cond_net = FeedForwardNet(input_dim=x_g.shape[1], hidden_dim=64, output_dim=1)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

sdf_net.apply(init_weights)
cond_net.apply(init_weights)

# Step 3: Optimizers
opt_sdf = torch.optim.Adam(sdf_net.parameters(), lr=1e-3)
opt_cond = torch.optim.Adam(cond_net.parameters(), lr=1e-3)

# GAN Loop Parameters
n_epochs = 100
inner_steps = 5  # Number of steps per GAN player before switching

for epoch in range(n_epochs):
    # ---- Step A: Maximize loss w.r.t. adversary (conditional network) ----
    for _ in range(inner_steps):
        omega = sdf_net(x_omega).squeeze()
        g = cond_net(x_g).squeeze().clamp(-5, 5)

        # SDF Calculation (no-arbitrage)
        M = 1 - (omega * y).sum()

        # Moment condition (pricing errors)
        moment_term = M * y * g
        loss_g = torch.mean(moment_term) ** 2

        # Optimize conditional network
        opt_cond.zero_grad()
        (-loss_g).backward()  # Maximize loss (minimax)
        opt_cond.step()

    # ---- Step B: Minimize loss w.r.t. SDF network ----
    for _ in range(inner_steps):
        omega = sdf_net(x_omega).squeeze()
        g = cond_net(x_g).squeeze().clamp(-5, 5)

        # SDF Calculation (no-arbitrage)
        M = 1 - (omega * y).sum()

        # Moment condition (pricing errors)
        moment_term = M * y * g
        loss_omega = torch.mean(moment_term) ** 2

        # Optimize SDF network
        opt_sdf.zero_grad()
        loss_omega.backward()
        opt_sdf.step()

    # Logging for stability & convergence diagnostics
    if epoch % 10 == 0 or epoch == n_epochs - 1:
        print(f"[Epoch {epoch:03}] ✅ Loss = {loss_omega.item():.6f}")
        print(f"  M = {M.item():.6f}, ω·R sum = {(omega * y).sum().item():.6f}")
        print(f"  ω.mean = {omega.mean().item():.4f}, g.mean = {g.mean().item():.4f}")


# Step 4: Extract final ω and g, attach to asset-level data and save

# Ensure eval mode before inference
sdf_net.eval()
cond_net.eval()

with torch.no_grad():
    omega_final = sdf_net(x_omega).squeeze().cpu().numpy()
    g_final = cond_net(x_g).squeeze().cpu().numpy()

# Retrieve and refresh merged panel from preprocessor
preprocessor = TensorPreprocessor(h_t, h_t_g, I_ti)
_, _, _ = preprocessor.preprocess()
panel_out = preprocessor.merged.copy()

# Add predictions to the panel
panel_out["omega"] = omega_final
panel_out["g"] = g_final

# ✅ Full panel with all features + omega/g
panel_out.to_csv("trained_outputs_full.csv", index=False)
print("[✓] Saved full panel to 'trained_outputs_full.csv'")

# ✅ Save just id_stock, date, omega, and g
panel_out[["date", "id_stock", "omega", "g"]].to_csv("trained_weights.csv", index=False)
print("[✓] Saved weights to 'trained_weights.csv'")

# Print grouped summaries
from pandas import qcut

print("\n[Decile Summary] Mean ω by MarketCap decile:")
print(panel_out.groupby(qcut(panel_out["MarketCap"], 10))["omega"].mean())

print("\n[Decile Summary] Mean g by ExcessRet decile:")
print(panel_out.groupby(qcut(panel_out["ExcessRet"], 10))["g"].mean())

# Top ω assets per month
top_omega_assets = (
    panel_out.sort_values("omega", ascending=False)
    .groupby("date")
    .head(5)[["date", "id_stock", "omega", "ExcessRet"]]
)

print("\n[Top ω Assets by Month]:")
print(top_omega_assets.head(10))
