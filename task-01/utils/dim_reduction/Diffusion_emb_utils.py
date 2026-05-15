import numpy as np
import warnings
from sklearn.neighbors import NearestNeighbors
from utils.riemmanian_geometry.Riemannian_utils import _riemannian_dist

np.random.seed(42)

def _make_row_stochastic(kernel):
    'This does not strictly mean row-stochastic,but normalizes the kernel'
    column_sum = np.sum(kernel, axis=0)
    row_stochastic_kernel = np.einsum("i, j, ij -> ij",
                                      1 / np.sqrt(column_sum),
                                      1 / np.sqrt(column_sum),
                                      kernel)
    return row_stochastic_kernel

def _regularize_by_median_sv(correlations, signal, window_length=0, subsampling=0):
    if subsampling > 0:
        length = signal.shape[-1]
        midpoints = np.linspace(window_length // 2,
                                length - window_length // 2,
                                subsampling)
        midpoints = list(map(int, midpoints))
        u, s, v = np.linalg.svd(signal[..., midpoints])
    else:
        u, s, v = np.linalg.svd(signal)
    regularizer = np.median(s, axis=1)[:, None, None] * np.eye(
        correlations.shape[-1])[None, :, :]
    return correlations + regularizer[:, None, :, :]

def _regularize_by_smallest_ev(correlations, eps=1e-3):
    eig = np.linalg.eigvals(correlations)
    for idx, e in enumerate(eig):
        if any(e.flatten() < 0):
            eps = eps - np.min(e.flatten())
            regularizer = eps * np.eye(correlations.shape[-1])
            correlations[idx] = regularizer[None, :, :] + correlations[idx]
    return correlations

def _get_kernel_riemannian(all_distances, ratio_nn=0.1, eps=2, use_sparse=True):
    """
    Build Riemannian diffusion kernel.

    Sigma is estimated from nearest-neighbor distances.
    If use_sparse=False: dense kernel - all pairwise distances get weights, using local sigma_i.
    If use_sparse=True: sparse kernel - only nearest-neighbor weights are kept.
    """
    all_distances = np.asarray(all_distances, dtype=float)
    K = all_distances.shape[0]

    nnb = max(2, int(np.floor(K * ratio_nn)))

    idx = np.argsort(all_distances, axis=1)[:, 1:nnb + 1]
    nn_d = np.take_along_axis(all_distances, idx, axis=1)

    sigma = np.median(nn_d, axis=1)
    sigma[sigma == 0] = 1e-12

    if not use_sparse:
        kernel = np.exp(- (all_distances / (np.sqrt(2) * eps * sigma[:, None])) ** 2)
    else:
        w = np.exp( - (nn_d / (np.sqrt(2) * eps * sigma[:, None])) ** 2)

        kernel = np.zeros((K, K), dtype=float)
        for i in range(K):
            kernel[i, idx[i]] = w[i]

    kernel = 0.5 * (kernel + kernel.T)
    kernel = _make_row_stochastic(kernel)

    return kernel

def _get_kernel_euclidean(X, scale_k):
    neighbors = NearestNeighbors(n_neighbors=scale_k, algorithm='auto').fit(X)
    nearest_distances, indices = neighbors.kneighbors(X)
    sigma = np.median(nearest_distances, axis=1)
    nonvanishing_entries = np.exp(- (nearest_distances / sigma[:, None]) ** 2)
    kernel = np.zeros(shape=(len(X), len(X)))
    for i in range(len(X)):
        kernel[i, indices[i, :]] = nonvanishing_entries[i, :]
    kernel = (kernel + kernel.T) / 2
    kernel = _make_row_stochastic(kernel)
    return kernel, nearest_distances

def get_diffusion_embedding(correlations, window_length, scale_k=20, signal=None, subsampling=0, mode='riemannian', return_kernel=False):
    """
    :param
    correlations: (Bx)KxNxN K correlation matrices. Will be carried out
    over all first dimensions
    :param
    scale_k: number of nearest neighbors to use for evaluating the scale
    :param
    tol: tolerance when iteration to get Riemannian mean converged
    :param
    maxiter: when to stop Riemannian mean algorithm
    :param
    vector_input: to use diffusion embedding onto vectors, not correlation
    matrices. Only used to test functionality. Use with care.
    :return:
    """

    if ((correlations.ndim == 3 and mode != 'vector_input') or
            (correlations.ndim == 2 and mode == 'vector_input')):
        correlations = np.array([correlations])
    elif ((correlations.ndim == 4 and mode != 'vector_input') or
          (correlations.ndim == 3 and mode == 'vector_input')):
        pass
    else:
        raise ValueError(f"correlations must be shape (Bx)KxNxN but is "
                         f"{correlations.shape}")

    if window_length < correlations.shape[-1] and mode != 'vector_input':
        warnings.warn("Small window_length. Regularizing correlations.")
        if signal is not None:
            if subsampling > 0:
                correlations = _regularize_by_median_sv(
                    correlations, signal, window_length, subsampling)
            else:
                correlations = _regularize_by_median_sv(correlations, signal)
        else:
            correlations = _regularize_by_smallest_ev(correlations)

    distances = []
    diffusion_representations = []
    for corrs in correlations:
        if mode == 'riemannian':
            dists = _riemannian_dist(corrs)
            distances.append(dists)
            kernel = _get_kernel_riemannian(dists)
        elif mode == 'euclidean':
            kernel, dists = _get_kernel_euclidean(
                corrs.reshape(corrs.shape[:-2] + (-1,)), scale_k)
            distances.append(dists)
        else:
            raise ValueError(f'{mode=}')
        if return_kernel:
            return kernel, np.array(distances)
        eig, vec = np.linalg.eigh(kernel)
        sort_idx = eig.argsort()[-2::-1]
        vec = vec.T[sort_idx]
        vec = eig[sort_idx, None] * vec

        diffusion_representations.append(vec)

    return np.array(diffusion_representations), np.array(distances)













