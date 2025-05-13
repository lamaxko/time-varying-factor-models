import torch
import torch.nn as nn
import torch.optim as optim

# Recurrent model to summarize macro states
class MacroStateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]  # return last time step

# Feedforward network for ω and g
class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Simulated setup
T, N = 100, 50  # Time steps, number of stocks
macro_features, char_features = 10, 15

# Dummy data
macro_data = torch.randn(T, macro_features)
print(macro_data)
stock_chars = torch.randn(T, N, char_features)
print(stock_chars)
stock_returns = torch.randn(T, N)
print(stock_returns)

# Model init
hidden_size = 4
state_rnn = MacroStateLSTM(macro_features, hidden_size)
moment_rnn = MacroStateLSTM(macro_features, hidden_size)
sdf_net = FeedForwardNet(hidden_size + char_features, 32, 1)
cond_net = FeedForwardNet(hidden_size + char_features, 32, 1)

# Optimizers
opt_sdf = optim.Adam(list(state_rnn.parameters()) + list(sdf_net.parameters()), lr=1e-3)
opt_cond = optim.Adam(list(moment_rnn.parameters()) + list(cond_net.parameters()), lr=1e-3)

# Training loop (5 epochs)
for epoch in range(5):
    # Prepare inputs
    macro_seq = macro_data.unsqueeze(0).repeat(N, 1, 1)
    print(macro_seq)
    h_sdf = state_rnn(macro_seq)
    print(h_sdf)
    h_cond = moment_rnn(macro_seq)
    print(h_cond)

    x_sdf = torch.cat([h_sdf, stock_chars[-1]], dim=1)
    print(x_sdf)
    x_cond = torch.cat([h_cond, stock_chars[-1]], dim=1)
    print(x_cond)

    omega = sdf_net(x_sdf).squeeze()
    g = cond_net(x_cond).squeeze()
    rets = stock_returns[-1]

    M = 1 - torch.sum(omega * rets)
    loss = (torch.mean(M * rets * g)) ** 2

    if epoch % 2 == 0:
        opt_cond.zero_grad()
        (-loss).backward()
        opt_cond.step()
    else:
        opt_sdf.zero_grad()
        loss.backward()
        opt_sdf.step()

    print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
