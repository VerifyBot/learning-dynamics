# Saxe Learning Dynamics Simulation

Simulations of learning dynamics in deep linear and non-linear neural networks, based on Andrew Saxe's exact analytical solutions. Visualizes representation drift using Euclidean and Riemannian Diffusion Maps.

## Architecture

```bash
src/
├── models.py         # PyTorch network classes (Linear, Tanh, Riemannian, RiemannianTanh, DriftReLU)
├── trainer.py        # Training loops (SaxeTrainer, DriftTrainer) + model registry
├── dataset.py        # Hierarchical binary-tree dataset generator
├── analysis.py       # Saxe's exact analytical mode-strength solutions
├── visualization.py  # All plotting (dynamics, delay, loss, trajectories, breakthrough)
└── log.py            # Logging setup (console INFO + per-run DEBUG file in logs/)

utils/
├── dim_reduction/    # Diffusion Map embedding (Euclidean & Riemannian kernels)
├── riemmanian_geometry/  # Riemannian distance computations for SPD matrices
└── visualization/    # Scatter/barplot helpers
```

## Adding a New Model

1. Create a class in `src/models.py` inheriting from `SaxeLinearNetwork`
2. Set class attributes: `label`, `color`, `linestyle`, `marker`, `uses_riemannian`
3. Append the class to `SAXE_MODEL_CLASSES` in `src/trainer.py`

## Setup & Usage

### Local (with uv)

```bash
uv sync
uv run python main.py              # main simulation
uv run python run_breakthrough.py   # drift experiment
```

### Docker

Build the image once (dependencies are cached in a separate layer):

```bash
docker build -t saxe-sim .
```

Run `main.py`

```bash
docker run --rm -v ./output:/app/output -v ./logs:/app/logs saxe-sim
```

Run `run_breakthrough.py`:

```bash
docker run --rm -v ./output:/app/output -v ./logs:/app/logs saxe-sim run_breakthrough.py
```

## Logging

- **Console** (INFO): Concise progress — phase starts, timings, final losses.
- **File** (DEBUG): Full epoch-level detail — per-model losses, Delta C checks, Hessian traces. Saved to `logs/<run_id>.log`.

## Outputs

All outputs are saved to `output/<timestamp>/`:
- `figure3_left_dynamics.png` — SVD mode strengths over epochs
- `loss_comparison.png` — loss convergence across all models
- `figure3_right_delay.png` — mode emergence delay relative to analytic
- `trajectory_<model>_<metric>.png` — Diffusion Map trajectories
- `cache/` — pickled intermediate results for fast re-runs
