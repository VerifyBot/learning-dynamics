import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import typing
from matplotlib.lines import Line2D

from src.trainer import SAXE_MODEL_CLASSES

try:
    from utils.visualization.utils_visualization import scatter_value_map, barplot_mean_sem
except ImportError:
    pass


def _get_model_styles() -> list[dict]:
    """Reads display metadata from the model registry so plots stay in sync with models."""
    return [
        {"label": cls.label, "color": cls.color, "linestyle": cls.linestyle, "marker": cls.marker}
        for cls in SAXE_MODEL_CLASSES
    ]


def plot_learning_dynamics(results: dict[str, typing.Any], ana_modes: np.ndarray,
                           out_path: str, title: str = "Figure 3: Learning Dynamics"):
    """
    Plots the staged learning dynamics of selected SVD modes over time.
    Automatically includes every model found in the results dict.

    @param results: Training results dict with "{label}_mode_strengths" keys.
    @param ana_modes: Analytically derived mode strengths (shape: [epochs, modes]).
    @param out_path: File path to save the plot.
    @param title: Plot title.
    """
    plt.figure(figsize=(10, 7))
    modes_to_plot = [0, 1, 2, 4, 11, 17, 30]
    styles = _get_model_styles()

    # Analytic baseline
    epochs = np.arange(ana_modes.shape[0])
    for i in modes_to_plot:
        plt.plot(epochs, ana_modes[:, i], color='red', linestyle='-', alpha=0.5)

    # Each model
    for style in styles:
        key = f"{style['label']}_mode_strengths"
        if key not in results:
            continue
        modes = results[key]
        for i in modes_to_plot:
            plt.plot(epochs, modes[:, i], color=style["color"],
                     linestyle=style["linestyle"], alpha=0.8)

    # Legend
    legend_lines = [Line2D([0], [0], color='red', lw=2)]
    legend_labels = ['Analytic']
    for style in styles:
        if f"{style['label']}_mode_strengths" in results:
            legend_lines.append(
                Line2D([0], [0], color=style["color"], linestyle=style["linestyle"], lw=2))
            legend_labels.append(style["label"])

    plt.legend(legend_lines, legend_labels)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Mode Strength')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_delay(delay_results: dict[str, np.ndarray], out_path: str):
    """
    Plots the delay of each model's mode emergence relative to the analytical expectation.

    @param delay_results: Dict mapping model label -> delay array (shape: [runs, modes]).
    @param out_path: File path to save the plot.
    """
    modes = np.arange(1, 33)
    styles = {s["label"]: s for s in _get_model_styles()}

    plt.figure(figsize=(10, 7))
    for label, delays in delay_results.items():
        mean = np.nanmean(delays, axis=0)
        std = np.nanstd(delays, axis=0)
        style = styles.get(label, {"color": "gray", "marker": "x"})
        plt.errorbar(modes, mean, yerr=std, fmt=f'{style["marker"]}-',
                     color=style["color"], label=label)

    plt.axhline(0, color='black', linestyle='--')
    plt.title("Figure 3: Delay in Learning")
    plt.xlabel('Input-Output Mode')
    plt.ylabel('Delay relative to analytic')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_trajectory(representations: np.ndarray, vals: np.ndarray, out_path: str, title: str):
    """Visualizes the Diffusion Map trajectory (2D or 3D) of the hidden representations."""
    try:
        from utils.visualization.utils_visualization import scatter_value_map
        if representations.shape[1] >= 3:
            scatter_value_map(representations[:, :3], values=vals, s=20, title=title,
                              xlabel="Dim 1", ylabel="Dim 2", zlabel="Dim 3",
                              colorbar_label="Epochs")
        else:
            scatter_value_map(representations[:, :2], values=vals, s=20, title=title,
                              xlabel="Dim 1", ylabel="Dim 2", colorbar_label="Epochs")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except ImportError:
        print("Warning: utils_visualization not found. Skipping plot_trajectory.")


def plot_loss_comparison(results: dict[str, typing.Any], out_path: str):
    """
    Compares the loss convergence (log scale) across all models that have loss data.
    """
    styles = {s["label"]: s for s in _get_model_styles()}

    plt.figure(figsize=(10, 7))
    for label, style in styles.items():
        key = f"{label}_loss"
        if key not in results:
            continue
        loss = results[key]
        epochs = np.arange(len(loss))
        plt.plot(epochs, loss, color=style["color"], label=label,
                 linewidth=2, linestyle=style["linestyle"])

    plt.yscale('log')
    plt.title("Convergence Comparison")
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss (Log Scale)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_aviv_breakthrough_panel(results: dict[str, typing.Any], diff_map_coords: np.ndarray,
                                 epochs_sampled: np.ndarray, out_path: str):
    """
    Generates a 3-panel breakthrough visualization detailing the drift phases:
    1. Fast Familiarity (Task Acquisition)
    2. Directed Drift (Implicit Regularization via Noise)
    3. Representational Drift on the Zero-Loss Manifold

    @param results: Dictionary containing 'loss', 'hessian_trace', and 'hidden_sparsity'.
    @param diff_map_coords: Riemannian Diffusion Map 2D coordinates (shape: [N, 2]).
    @param epochs_sampled: Epoch indices corresponding to the logged metrics.
    @param out_path: File path to save the panel.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=False)

    ax1 = axes[0]
    ax1.plot(epochs_sampled, results['loss'], color='black', lw=2)
    ax1.set_yscale('log')
    ax1.set_ylabel('MSE Loss (Log)')
    ax1.set_title('Phase 1: Fast Familiarity (Task Acquisition)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Epochs')

    ax2 = axes[1]
    color = 'tab:red'
    ax2.set_ylabel('Hessian Trace (Curvature)', color=color)
    ax2.plot(epochs_sampled, results['hessian_trace'], color=color, lw=2)
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax2.twinx()
    color = 'tab:blue'
    ax3.set_ylabel('Sparsity (% Dead Neurons)', color=color)
    ax3.plot(epochs_sampled, results['hidden_sparsity'], color=color, lw=2, linestyle='--')
    ax3.tick_params(axis='y', labelcolor=color)
    ax2.set_title('Phase 2: Directed Drift (Implicit Regularization via Noise)')
    ax2.set_xlabel('Epochs')

    ax4 = axes[2]
    scatter = ax4.scatter(diff_map_coords[:, 0], diff_map_coords[:, 1],
                          c=epochs_sampled, cmap='viridis', s=40, zorder=2)
    ax4.plot(diff_map_coords[:, 0], diff_map_coords[:, 1],
             color='gray', alpha=0.5, lw=1.5, zorder=1)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Epochs')
    ax4.set_xlabel('Diffusion Dimension 1')
    ax4.set_ylabel('Diffusion Dimension 2')
    ax4.set_title('Phase 3: Representational Drift on the Zero-Loss Manifold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
