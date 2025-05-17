import torch
import torch.nn as nn
from models.ffn import FeedForwardNet

class SDFGanArbitrage:
    def __init__(self, input_dim_omega, hidden_dim=64, lr=1e-3):
        self.sdf_net = FeedForwardNet(input_dim=input_dim_omega, hidden_dim=hidden_dim, output_dim=1)
        self._init_weights(self.sdf_net)
        self.output_transform = nn.Softplus()
        self.opt_sdf = torch.optim.Adam(self.sdf_net.parameters(), lr=lr)

    def _init_weights(self, model):
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.1)

    def fit(self, x_omega, y, n_epochs=100, inner_steps=5, group_size=500):
        early_stop_counter = 0
        for epoch in range(n_epochs):
            for _ in range(inner_steps):
                raw_omega = self.sdf_net(x_omega).squeeze()
                omega = self.output_transform(raw_omega)

                # Reshape and normalize per group (e.g., per month)
                if omega.dim() == 1:
                    omega = omega.unsqueeze(1)

                n = omega.shape[0]
                pad = (group_size - n % group_size) % group_size
                if pad > 0:
                    omega = torch.cat([omega, torch.zeros(pad, 1, device=omega.device)], dim=0)
                    y = torch.cat([y, torch.zeros(pad, device=y.device)], dim=0)

                omega_grouped = omega.view(-1, group_size)
                omega_normalized = omega_grouped / (omega_grouped.sum(dim=1, keepdim=True) + 1e-8)
                omega = omega_normalized.view(-1)

                # Cut padded values
                omega = omega[:n]
                y = y[:n]

                # Maximize expected return
                loss_omega = -torch.sum(omega * y) + 1e-5 * torch.sum(omega**2)  # L2 regularization

                self.opt_sdf.zero_grad()
                loss_omega.backward()
                self.opt_sdf.step()

            print(f"Epoch {epoch:03d} | Loss: {loss_omega.item():.6f}")
            print(f"  Sum(ω·R):             {(omega * y).sum().item():.6f}")
            print(f"  ω mean (SDF weights): {omega.mean().item():.4f}")

            if loss_omega.item() > -1e-4:
                early_stop_counter += 1
                if early_stop_counter >= 5:
                    print(f"[Early Stop] Low gain – stopping at epoch {epoch:03d}")
                    break
            else:
                early_stop_counter = 0

    def extract_outputs(self, x_omega, group_size=500):
        self.sdf_net.eval()
        with torch.no_grad():
            raw_omega = self.sdf_net(x_omega).squeeze()
            omega = self.output_transform(raw_omega)

            n = omega.shape[0]
            pad = (group_size - n % group_size) % group_size
            if pad > 0:
                omega = torch.cat([omega, torch.zeros(pad, device=omega.device)], dim=0)

            omega_grouped = omega.view(-1, group_size)
            omega_normalized = omega_grouped / (omega_grouped.sum(dim=1, keepdim=True) + 1e-8)
            omega = omega_normalized.view(-1)[:n]

        return omega.cpu().numpy()
