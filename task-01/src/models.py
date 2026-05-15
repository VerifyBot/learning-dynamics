import torch
import torch.nn as nn
import numpy as np

class SaxeLinearNetwork(nn.Module):
    """
    Deep Linear Network used to test Andrew Saxe's learning dynamics.

    This network has no non-linearities and trains directly on the SVD modes
    of the dataset.
    """
    # Display metadata used by visualization and main loops.
    label = "Linear"
    color = "blue"
    linestyle = "--"
    marker = "o"
    # Whether this model needs Riemannian preconditioning on its upper-layer gradient.
    uses_riemannian = False

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, u0: float = 1e-3):
        """
        @param input_dim: Dimensionality of the input (number of leaves).
        @param hidden_dim: Dimensionality of the hidden bottleneck.
        @param output_dim: Dimensionality of the output (number of features).
        @param u0: Initial strength of the mode.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self._init_weights(u0)

    def _init_weights(self, u0: float):
        """Applies Saxe's initialization scheme so initial modes are roughly `u0`."""
        std = np.sqrt(u0) / 2
        with torch.no_grad():
            self.fc1.weight.copy_(torch.randn_like(self.fc1.weight) * std)
            self.fc2.weight.copy_(torch.randn_like(self.fc2.weight) * std)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        @param x: Input tensor.
        @return: (hidden_activations, output_predictions).
        """
        h = self.fc1(x)
        out = self.fc2(h)
        return h, out

    def is_linear_model(self) -> bool:
        """Whether the effective weight matrix W_eff = W2 @ W1 fully describes the model output."""
        return True


class SaxeTanhNetwork(SaxeLinearNetwork):
    """
    Non-linear network using Tanh activation.

    Starts in the linear regime (since tanh(x) ~ x for small x) and diverges
    as the weights grow and the non-linearity activates.
    """
    label = "Tanh"
    color = "green"
    linestyle = "-."
    marker = "s"

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, u0: float = 1e-3):
        super().__init__(input_dim, hidden_dim, output_dim, u0)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_pre = self.fc1(x)
        h_act = self.activation(h_pre)
        out = self.fc2(h_act)
        return h_act, out

    def is_linear_model(self) -> bool:
        return False


class RiemannianLinearNetwork(SaxeLinearNetwork):
    """
    Deep Linear Network trained with Natural Gradient (Riemannian preconditioning).

    Same architecture as SaxeLinearNetwork, but the trainer applies Riemannian
    preconditioning on the upper-layer gradient before stepping.
    """
    label = "Riemannian"
    color = "purple"
    linestyle = ":"
    marker = "d"
    uses_riemannian = True


class RiemannianTanhNetwork(SaxeTanhNetwork):
    """
    Tanh network trained with Natural Gradient (Riemannian preconditioning).

    Combines the Tanh non-linearity with Riemannian preconditioning.
    The preconditioning is computed from the hidden-layer correlation C = h @ h^T
    (post-activation), so the metric adapts to the non-linear representation geometry.
    """
    label = "Riemannian+Tanh"
    color = "orange"
    linestyle = (0, (3, 1, 1, 1))  # dash-dot-dot
    marker = "^"
    uses_riemannian = True


class DriftReluNetwork(SaxeLinearNetwork):
    """
    Non-linear network using ReLU.

    Used to demonstrate topological sparsification (dead neurons) via representational drift.
    """
    label = "DriftReLU"
    color = "red"
    linestyle = "-"
    marker = "v"

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, u0: float = 1e-3):
        super().__init__(input_dim, hidden_dim, output_dim, u0)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_pre = self.fc1(x)
        h_act = self.activation(h_pre)
        out = self.fc2(h_act)
        return h_act, out

    def is_linear_model(self) -> bool:
        return False
