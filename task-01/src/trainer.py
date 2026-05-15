import logging
import torch
import torch.nn as nn
import numpy as np
import typing

from src.models import (
    SaxeLinearNetwork, SaxeTanhNetwork,
    RiemannianLinearNetwork, RiemannianTanhNetwork,
    DriftReluNetwork,
)
from src.log import Timer

log = logging.getLogger("saxe")

# Registry of model classes used in the main Saxe comparison.
# To add a new model, just append its class here.
SAXE_MODEL_CLASSES: list[type[SaxeLinearNetwork]] = [
    SaxeLinearNetwork,
    SaxeTanhNetwork,
    RiemannianLinearNetwork,
    RiemannianTanhNetwork,
]


class SaxeTrainer:
    """
    Trains all registered Saxe model variants side-by-side with identical
    initialization and compares their learning dynamics.

    The Riemannian preconditioning is applied per-model based on each model's
    `uses_riemannian` flag. For non-linear models (Tanh), the preconditioning
    metric C is computed from the post-activation hidden correlations (h @ h^T),
    which means the geometry adapts to the actual representation.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, U: np.ndarray, V: np.ndarray,
                 epochs: int = 1500, lr: float = 0.5e-3, u0: float = 1e-3,
                 seed: int | None = None):
        """
        @param X: Input dataset (shape: [num_leaves, num_leaves]).
        @param Y: Target dataset (shape: [num_features, num_leaves]).
        @param U: Left singular vectors of Y.
        @param V: Right singular vectors of Y.
        @param epochs: Number of training epochs.
        @param lr: Learning rate.
        @param u0: Mode strength initialization scale.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.epochs = epochs
        self.lr = lr

        self.X_t = torch.tensor(X.T, dtype=torch.float32)
        self.Y_t = torch.tensor(Y.T, dtype=torch.float32)
        self.U_t = torch.tensor(U, dtype=torch.float32)
        self.V_t = torch.tensor(V, dtype=torch.float32)

        input_dim, hidden_dim, output_dim = X.shape[0], X.shape[0], Y.shape[0]

        # Build all models and sync their initial weights from the first model.
        self.models: dict[str, SaxeLinearNetwork] = {}
        self.optimizers: dict[str, torch.optim.SGD] = {}
        ref_model = None

        for cls in SAXE_MODEL_CLASSES:
            model = cls(input_dim, hidden_dim, output_dim, u0)
            if ref_model is None:
                ref_model = model
            else:
                with torch.no_grad():
                    model.fc1.weight.copy_(ref_model.fc1.weight)
                    model.fc2.weight.copy_(ref_model.fc2.weight)
            self.models[cls.label] = model
            self.optimizers[cls.label] = torch.optim.SGD(model.parameters(), lr=lr)

        log.debug(f"SaxeTrainer initialized: {len(self.models)} models, "
                  f"{epochs} epochs, lr={lr}, u0={u0}")

    def _apply_riemannian_preconditioning(self, model: SaxeLinearNetwork, h: torch.Tensor):
        """
        Warps the Euclidean gradient on W2 into the Natural Gradient direction.

        For linear models: C = W1 @ W1^T (weight-space metric).
        For non-linear models: C = h^T @ h (post-activation metric).
        """
        with torch.no_grad():
            if model.is_linear_model():
                C = model.fc1.weight @ model.fc1.weight.T
            else:
                C = (h.T @ h) / h.shape[0]

            max_eig = torch.max(torch.linalg.eigvalsh(C))
            damping = 0.05 * max_eig + 1e-5
            C_stable = C + damping * torch.eye(C.shape[0], device=C.device)
            C_inv = torch.linalg.inv(C_stable)

            grad_orig = model.fc2.weight.grad.clone()
            grad_riem = grad_orig @ C_inv

            scale = torch.norm(grad_orig) / (torch.norm(grad_riem) + 1e-8)
            model.fc2.weight.grad = grad_riem * scale

    def _compute_roei_delta_c(self, W21_old: torch.Tensor, W32_old: torch.Tensor) -> torch.Tensor:
        """
        Computes the theoretical first-order discrete gradient step for covariance C.
        """
        Y_mat = self.Y_t.T
        error_matrix = Y_mat - (W32_old @ W21_old)
        term1 = W32_old.T @ error_matrix @ W21_old.T
        term2 = W21_old @ error_matrix.T @ W32_old
        return self.lr * (term1 + term2)

    def _get_mode_strength(self, model: SaxeLinearNetwork,
                           Y_pred: torch.Tensor) -> torch.Tensor:
        """
        Extracts SVD mode strengths. Uses W_eff for linear models,
        projects Y_pred onto SVD basis for non-linear ones.
        """
        if model.is_linear_model():
            W_eff = model.fc2.weight @ model.fc1.weight
            return torch.diag(self.U_t.T @ W_eff @ self.V_t)
        else:
            return torch.diag(self.U_t.T @ Y_pred.T @ self.V_t)

    def train(self, track_correlations: bool = True) -> dict[str, typing.Any]:
        """
        Runs training for all registered models in lock-step.

        @param track_correlations: If True, records hidden correlations per epoch.
        @return: Dict keyed by "{label}_mode_strengths", "{label}_loss",
                 and optionally "{label}_hidden_correlations".
        """
        results: dict[str, list] = {}
        for label in self.models:
            results[f"{label}_mode_strengths"] = []
            results[f"{label}_loss"] = []
            if track_correlations:
                results[f"{label}_hidden_correlations"] = []

        lin_model = self.models.get("Linear")
        log_interval = max(1, self.epochs // 5)  # log ~5 times during training

        with Timer(f"Training {len(self.models)} models × {self.epochs} epochs"):
            for epoch in range(self.epochs):
                # --- Delta C verification (linear model only) ---
                if lin_model is not None:
                    with torch.no_grad():
                        W21_old = lin_model.fc1.weight.clone()
                        W32_old = lin_model.fc2.weight.clone()
                        C_old = W21_old @ W21_old.T
                        delta_C_roei = self._compute_roei_delta_c(W21_old, W32_old)

                # --- Forward / backward / step for every model ---
                hidden_activations: dict[str, torch.Tensor] = {}

                for label, model in self.models.items():
                    opt = self.optimizers[label]
                    opt.zero_grad()
                    h, Y_pred = model(self.X_t)
                    loss = 0.5 * torch.sum((Y_pred - self.Y_t) ** 2)
                    loss.backward()

                    if model.uses_riemannian:
                        self._apply_riemannian_preconditioning(model, h)

                    opt.step()
                    hidden_activations[label] = h.detach()

                    with torch.no_grad():
                        mode = self._get_mode_strength(model, Y_pred)
                        results[f"{label}_mode_strengths"].append(mode)
                        results[f"{label}_loss"].append(loss.item())

                # --- Delta C check ---
                if lin_model is not None and epoch % log_interval == 0:
                    with torch.no_grad():
                        W21_new = lin_model.fc1.weight.clone()
                        C_new = W21_new @ W21_new.T
                        delta_C_empirical = C_new - C_old
                        max_diff = torch.max(torch.abs(delta_C_empirical - delta_C_roei)).item()
                        log.debug(f"Epoch {epoch}/{self.epochs} | "
                                  f"Delta C max diff: {max_diff:.2e}")

                # --- Progress log ---
                if epoch % log_interval == 0:
                    losses_str = ", ".join(
                        f"{lbl}={results[f'{lbl}_loss'][-1]:.4f}"
                        for lbl in self.models
                    )
                    log.debug(f"Epoch {epoch}/{self.epochs} | {losses_str}")

                # --- Track correlations ---
                if track_correlations:
                    with torch.no_grad():
                        for label in self.models:
                            h = hidden_activations[label]
                            corr = torch.nan_to_num(torch.corrcoef(h.T), nan=0.0)
                            results[f"{label}_hidden_correlations"].append(corr)

        # --- Finalize: stack tensors -> numpy ---
        for key in list(results.keys()):
            val = results[key]
            if not val:
                continue
            if isinstance(val[0], torch.Tensor):
                results[key] = torch.stack(val).numpy()
            else:
                results[key] = np.array(val)

        # Log final losses
        for label in self.models:
            final_loss = results[f"{label}_loss"][-1]
            log.info(f"  {label:20s} final loss: {final_loss:.6f}")

        return results


class DriftTrainer:
    """
    Handles training for models experiencing representational drift due to noise injection.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, epochs: int = 5000,
                 lr: float = 0.5e-3, noise_std: float = 1e-4, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.epochs = epochs
        self.noise_std = noise_std

        self.X_t = torch.tensor(X.T, dtype=torch.float32)
        self.Y_t = torch.tensor(Y.T, dtype=torch.float32)

        self.model = DriftReluNetwork(32, 32, 1000, u0=1e-3)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        log.debug(f"DriftTrainer initialized: {epochs} epochs, lr={lr}, noise_std={noise_std}")

    def _hutchinson_trace(self, loss: torch.Tensor, num_samples: int = 5) -> float:
        """Approximates the trace of the Hessian using Hutchinson's stochastic estimator."""
        trace = 0.0
        params = [p for p in self.model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

        for _ in range(num_samples):
            v = [torch.randint_like(p, high=2) * 2 - 1.0 for p in params]
            vjp = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
            trace += sum(torch.sum(vj * vi).item() for vj, vi in zip(vjp, v))

        return trace / num_samples

    def train(self) -> dict[str, typing.Any]:
        """Executes the drift training loop with noise injection."""
        results: dict[str, list] = {
            'loss': [], 'hessian_trace': [], 'hidden_sparsity': [],
            'hidden_correlations': [], 'weights_W21': [],
        }
        log_interval = max(50, self.epochs // 10)

        with Timer(f"Drift training × {self.epochs} epochs"):
            for epoch in range(self.epochs):
                self.optimizer.zero_grad()
                h_act, Y_pred = self.model(self.X_t)
                loss = 0.5 * torch.sum((Y_pred - self.Y_t) ** 2)

                if epoch % 50 == 0:
                    with torch.no_grad():
                        sparsity = (h_act == 0).float().mean().item()
                        corr = torch.nan_to_num(torch.corrcoef(h_act.T), nan=0.0)

                        results['hidden_correlations'].append(corr.numpy())
                        results['hidden_sparsity'].append(sparsity)
                        results['loss'].append(loss.item())
                        results['weights_W21'].append(self.model.fc1.weight.clone().numpy())

                    htrace = self._hutchinson_trace(loss)
                    results['hessian_trace'].append(htrace)

                    if epoch % log_interval == 0:
                        log.debug(f"Drift epoch {epoch}/{self.epochs} | "
                                  f"Loss: {loss.item():.4f} | Sparsity: {sparsity:.2%} | "
                                  f"Hessian Tr: {htrace:.4f}")

                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    for param in self.model.parameters():
                        param.add_(torch.randn_like(param) * self.noise_std)

        results['hidden_correlations'] = np.array(results['hidden_correlations'])
        results['weights_W21'] = np.array(results['weights_W21'])
        log.info(f"  Drift final loss: {results['loss'][-1]:.6f}")
        return results
