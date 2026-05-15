import logging
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (works in Docker / headless)
import matplotlib.pyplot as plt
import pickle
import torch
import multiprocessing as mp
from datetime import datetime
from functools import partial

from src.log import setup_logging, Timer
from src.dataset import generate_hierarchical_dataset
from src.trainer import SaxeTrainer, SAXE_MODEL_CLASSES
from src.analysis import analytical_solution
from src.visualization import (
    plot_learning_dynamics, plot_delay, plot_trajectory, plot_loss_comparison,
)

from utils.dim_reduction.Diffusion_emb_utils import get_diffusion_embedding

d = os.path.dirname(os.path.abspath(__file__))


def save_cache(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_cache(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None


def compute_half_time(mode_strengths: np.ndarray, max_val_per_mode: np.ndarray) -> np.ndarray:
    """
    For each SVD mode, finds the first epoch where the mode strength exceeds
    half of its theoretical maximum (the singular value s_alpha).
    """
    _, n_modes = mode_strengths.shape
    t_halfs = np.zeros(n_modes)
    for i in range(n_modes):
        s_max = max_val_per_mode[i]
        idx = np.where(mode_strengths[:, i] > s_max / 2)[0]
        t_halfs[i] = idx[0] if len(idx) > 0 else np.nan
    return t_halfs


def run_single_simulation(seed_idx, X, Y, U, V, epochs, lr, u0, S, t_half_analy):
    """Worker function for multiprocessing. Returns a dict of delays per model label."""
    torch.set_num_threads(1)

    trainer = SaxeTrainer(X, Y, U, V, epochs=epochs, lr=lr, u0=u0, seed=seed_idx)
    res = trainer.train(track_correlations=False)

    delays = {}
    for cls in SAXE_MODEL_CLASSES:
        key = f"{cls.label}_mode_strengths"
        if key in res:
            t_half = compute_half_time(res[key], max_val_per_mode=S)
            delays[cls.label] = (t_half - t_half_analy) / t_half_analy

    return delays


def create_main_run(use_cache=True, time_str=None):
    if time_str is None:
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log = setup_logging(run_id=time_str)

    out_dir = os.path.join(d, "output", time_str)
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    log.info(f"Run started — output: {out_dir}")
    log.debug(f"Models registered: {[c.label for c in SAXE_MODEL_CLASSES]}")

    # --- Dataset ---
    dataset_cache_path = os.path.join(cache_dir, "dataset.pkl")
    if use_cache and os.path.exists(dataset_cache_path):
        log.info("Using cached dataset")
        X, Y, U, S, V = load_cache(dataset_cache_path)
    else:
        with Timer("Dataset generation"):
            X, Y, U, S, V = generate_hierarchical_dataset(depth=5, num_features=1000, seed=42)
            save_cache((X, Y, U, S, V), dataset_cache_path)

    # --- 1. Single training run ---
    single_run_cache_path = os.path.join(cache_dir, "single_run.pkl")
    if use_cache and os.path.exists(single_run_cache_path):
        log.info("Using cached single training run")
        res_single = load_cache(single_run_cache_path)
    else:
        log.info("Starting single training run (1500 epochs, correlations=True)")
        trainer = SaxeTrainer(X, Y, U, V, epochs=1500, lr=0.5e-3, u0=1e-3, seed=42)
        res_single = trainer.train(track_correlations=True)
        save_cache(res_single, single_run_cache_path)

    ana = analytical_solution(S, epochs=1500, lr=0.5e-3, u0=1e-3)

    with Timer("Plotting dynamics + loss comparison"):
        plot_learning_dynamics(res_single, ana,
                               os.path.join(out_dir, "figure3_left_dynamics.png"))
        plot_loss_comparison(res_single, os.path.join(out_dir, "loss_comparison.png"))

    # --- 2. Diffusion Map trajectories ---
    dim_red_cache_path = os.path.join(cache_dir, "dim_red.pkl")
    if use_cache and os.path.exists(dim_red_cache_path):
        log.info("Using cached diffusion map embeddings")
        diff_results = load_cache(dim_red_cache_path)
    else:
        log.info("Computing Diffusion Map embeddings")
        diff_results = {}
        try:
            for cls in SAXE_MODEL_CLASSES:
                corr_key = f"{cls.label}_hidden_correlations"
                if corr_key not in res_single:
                    continue
                corr = res_single[corr_key]
                for dist_mode in ("euclidean", "riemannian"):
                    with Timer(f"Dim-Red: {cls.label} / {dist_mode}", level=logging.DEBUG):
                        emb, _ = get_diffusion_embedding(
                            corr, window_length=corr.shape[-1], scale_k=5, mode=dist_mode)
                        diff_results[f"{cls.label}_{dist_mode}"] = emb
            save_cache(diff_results, dim_red_cache_path)
        except Exception as e:
            log.error(f"Dimensionality reduction failed: {e}")
            diff_results = {}

    if diff_results:
        with Timer("Plotting trajectories"):
            epochs_array = np.arange(1500)
            for emb_key, emb in diff_results.items():
                plot_trajectory(
                    emb[0].T, epochs_array,
                    os.path.join(out_dir, f"trajectory_{emb_key}.png"),
                    title=emb_key.replace("_", " ").title(),
                )
        log.info(f"  {len(diff_results)} trajectory plots saved")

    # --- 3. Delay experiment (100 runs) ---
    delay_cache_path = os.path.join(cache_dir, "delay_results.pkl")
    if use_cache and os.path.exists(delay_cache_path):
        log.info("Using cached delay simulation results")
        delay_results = load_cache(delay_cache_path)
    else:
        n_runs = 100
        log.info(f"Starting delay experiment ({n_runs} parallel simulations)")

        t_half_analy = compute_half_time(ana, max_val_per_mode=S)
        worker_func = partial(run_single_simulation, X=X, Y=Y, U=U, V=V,
                              epochs=1500, lr=0.5e-3, u0=1e-3, S=S,
                              t_half_analy=t_half_analy)

        delay_lists: dict[str, list] = {cls.label: [] for cls in SAXE_MODEL_CLASSES}

        with Timer(f"Delay experiment ({n_runs} runs)"):
            with mp.get_context('spawn').Pool(processes=mp.cpu_count()) as pool:
                for i, delays in enumerate(pool.imap_unordered(worker_func, range(n_runs)), 1):
                    for label, d_arr in delays.items():
                        delay_lists[label].append(d_arr)
                    if i % 25 == 0 or i == n_runs:
                        log.info(f"  Delay progress: {i}/{n_runs}")

        delay_results = {label: np.array(arrs) for label, arrs in delay_lists.items()}
        save_cache(delay_results, delay_cache_path)

    with Timer("Plotting delay"):
        plot_delay(delay_results, os.path.join(out_dir, "figure3_right_delay.png"))

    log.info("Pipeline complete")


if __name__ == "__main__":
    create_main_run(use_cache=False)
