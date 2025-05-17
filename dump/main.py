import torch
import pandas as pd

from models.lstm import MacroStateLSTM
from models.ffn import FeedForwardNet
from apis.fred import FredApi
from preprocessing.stock_factors import FactorCalc
from preprocessing.tensors import TensorPreprocessor

# Step 1. Obtain DF of hidden states from two macro state and moment LSTM outputs
macro_stationary = FredApi().data_stationary.copy()

state_rnn = MacroStateLSTM(num_states=4, hidden_size=32, seq_len=12)
h_t = state_rnn.extract_from_dataframe(macro_stationary)

moment_rnn = MacroStateLSTM(num_states=4, hidden_size=32, seq_len=12)
h_t_g = moment_rnn.extract_from_dataframe(macro_stationary)

# Step 2. Obtain DF of Factors and Excess Return for each Asset
I_ti = FactorCalc().panel.copy()

# Step 3. Preprocess inputs into Tensors appropriate for FFN and GAN
x_omega, x_g, y = TensorPreprocessor(h_t, h_t_g, I_ti).preprocess()

# Step 4. Initialize the SDF network (ω) and the adversarial network (g)
input_dim_omega = x_omega.shape[1]
input_dim_g = x_g.shape[1]

sdf_net = FeedForwardNet(input_dim=input_dim_omega, hidden_dim=64, output_dim=1)
cond_net = FeedForwardNet(input_dim=input_dim_g, hidden_dim=64, output_dim=1)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

sdf_net.apply(init_weights)
cond_net.apply(init_weights)

# Step 5. Two separate optimizers
opt_sdf = torch.optim.Adam(sdf_net.parameters(), lr=1e-3)
opt_cond = torch.optim.Adam(cond_net.parameters(), lr=1e-3)

# Step 6. Run GAN Network

n_steps = 100  # Number of training steps

for step in range(n_steps):
    # ----- Forward pass -----
    # ω(h_t, I_{t,i}) from SDF network
    omega = sdf_net(x_omega).squeeze()

    # g(h_t^g, I_{t,i}) from conditional (adversary) network
    g = cond_net(x_g).squeeze()

    # clip extreme values early in training for stability
    # omega = torch.clamp(omega, 0, 5)
    g = torch.clamp(g, -5, 5)

    # Construct stochastic discount factor (scalar)
    # Equation (8) from paper:
    # M_{t+1} = 1 - ∑_i ω(h_t, I_{t,i}) * R^e_{t+1,i}
    weighted_returns = omega * y
    M = 1 - weighted_returns.sum()

    # ----- Loss calculation -----
    # Paper (Eq. 9):
    #   L(ω, g) = [ E[ M_{t+1} * R^e_{t+1,i} * g(h_t^g, I_{t,i}) ] ]²
    moment_term = M * y * g
    loss = torch.mean(moment_term) ** 2

    # ----- Alternating adversarial training -----
    if step % 2 == 0:
        # Maximize loss w.r.t. g (conditional moment network)
        opt_cond.zero_grad()
        (-loss).backward()
        opt_cond.step()
    else:
        # Minimize loss w.r.t. ω (SDF network)
        opt_sdf.zero_grad()
        loss.backward()
        opt_sdf.step()

    # ----- Logging -----
    # if step % 10 == 0 or step == n_steps - 1:
    print(f"[Step {step:03}] ✅ Loss = {loss.item():.6f}")
    print(f"        M = {M.item():.6f}, ω·R sum = {weighted_returns.sum().item():.6f}")
    print(f"        ω.mean = {omega.mean().item():.4f}, g.mean = {g.mean().item():.4f}")
