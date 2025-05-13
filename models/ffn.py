import torch
import torch.nn as nn

class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, dropout=0.0):
        """
        Generic Feedforward Network used for:
        - SDF network (ω)
        - Conditional moment network (g)

        Parameters:
        - input_dim: Dimension of macro + asset input
        - hidden_dim: Size of hidden layer
        - output_dim: Usually 1
        - dropout: Optional dropout for regularization
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# if __name__ == "__main__":
    # input_tensor = torch.cat([macro_states, asset_chars], dim=1)  # shape: (N, 19)
    # ffn = FeedForwardNet(input_dim=input_tensor.shape[1], hidden_dim=64)

    # output = ffn(input_tensor)  # shape: (N, 1)

