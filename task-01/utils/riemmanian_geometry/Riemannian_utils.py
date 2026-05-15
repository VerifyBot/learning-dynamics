import numpy as np
import itertools
from multiprocessing import Pool, cpu_count
from scipy.linalg import eigh, svd, logm, expm, pinv
from scipy.sparse import eye
import warnings

np.random.seed(42)

def sym_pos_def_dist(A, B, p=2):
    eig = np.linalg.eigvals(np.linalg.inv(A) @ B)
    # Ensure real, positive eigenvalues to avoid log(0) or complex warnings
    eig = np.maximum(np.abs(np.real(eig)), 1e-12)
    if p == 1:
        dist = np.sum(np.abs(np.log(eig)))
    else:
        dist = np.sum(np.abs(np.log(eig)) ** p) ** (1 / p)
    return dist

def _pairwise_dist_worker(args):
    """
    Worker function to calculate the distance between a single pair of matrices.
    Designed to run in parallel.
    """
    i, j, eig_A, vec_A, eig_B, vec_B, k = args
    
    # SVD of the interaction between the two filtered eigenvector bases
    try:
        OA, S, OB_Vh = np.linalg.svd(vec_A.T @ vec_B)
    except np.linalg.LinAlgError:
        OA, S, OB_Vh = svd(vec_A.T @ vec_B, lapack_driver='gesvd')
        
    # Clip S to prevent floating point issues with arccos
    S = np.clip(S, -1.0, 1.0)
    dU = np.linalg.norm(np.arccos(S))
    
    # MATHEMATICAL SHORTCUT:
    # Instead of reconstructing UA and doing dense UA.T @ A @ UA, 
    # we know vec_A.T @ A @ vec_A is just the diagonal matrix of eig_A.
    # Therefore, RA = OA.T @ diag(eig_A) @ OA. Same for RB.
    RA = OA.T @ (eig_A[:, None] * OA)
    RB = OB_Vh @ (eig_B[:, None] * OB_Vh.T)
    
    # Ensure symmetry
    RA = (RA + RA.T) / 2.0
    RB = (RB + RB.T) / 2.0
    
    dR = sym_pos_def_dist(RA, RB, p=2)
    
    d = np.sqrt(dU ** 2 + k * dR ** 2)
    return i, j, d

def _riemannian_dist(corrs, eigval_bound=0.01, k_val=1):
    """
    Optimized Riemannian distance calculation using multiprocessing and precomputation.
    """
    from multiprocessing import Pool, cpu_count
    import itertools
    import time
    N = len(corrs)
    
    # 1. Calculate the minimum rank 'r' across all matrices based on the threshold
    eig_counts = np.sum(np.linalg.eigvals(corrs) > eigval_bound, axis=1)
    r = np.min(eig_counts)
    
    print(f"      [_riemannian_dist] Rank r={r}, preparing {N} matrices...")
    start_t = time.time()
    
    # 2. Precompute eigenvalues and eigenvectors for ALL matrices upfront
    sym_corrs = (corrs + corrs.transpose(0, 2, 1)) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym_corrs)
    
    # Extract the top 'r' eigenvalues and eigenvectors
    top_eigvals = np.real(eigvals[:, -r:])
    top_eigvecs = np.real(eigvecs[:, :, -r:])
    
    # 3. Create arguments for parallel processing (upper triangle of the distance matrix)
    pairs = itertools.combinations(range(N), 2)
    tasks = ((i, j, top_eigvals[i], top_eigvecs[i], top_eigvals[j], top_eigvecs[j], k_val) for i, j in pairs)
    total_tasks = (N * (N - 1)) // 2
    
    # 4. Spin up the CPU pool and map the computations
    dR = np.zeros((N, N))
    n_cores = cpu_count()
    
    print(f"      [_riemannian_dist] Starting {total_tasks} parallel jobs on {n_cores} cores...", flush=True)
    
    with Pool(processes=n_cores) as pool:
        for count, (i, j, d) in enumerate(pool.imap_unordered(_pairwise_dist_worker, tasks, chunksize=2000), 1):
            dR[i, j] = d
            dR[j, i] = d  # Matrix is symmetric
            if count % 100000 == 0:
                print(f"      [_riemannian_dist] Progress: {count}/{total_tasks} pairs processed...", flush=True)
                
    end_t = time.time()
    print(f"      [_riemannian_dist] Completed in {end_t - start_t:.1f} seconds.")

    return dR

def safe_corr(x, y):
    # Ensure inputs are numpy arrays
    x, y = np.asarray(x), np.asarray(y)

    # Check for constant vectors (std = 0) and prevent division by zero
    if np.std(x) == 0 or np.std(y) == 0:
        return 0  # Correlation is undefined, return 0 or NaN

    # Compute correlation
    corr_matrix = np.corrcoef(x, y)

    return corr_matrix[0, 1]
def get_corr_matrix(matrix):
    """
    Computes the correlation matrix using fully vectorized NumPy operations.
    Replaces the slow nested loops and handles zero-variance (NaNs) natively.
    """
    # Ensure input is a numpy array
    matrix = np.asarray(matrix)
    
    # np.corrcoef natively handles 2D arrays (num_traces x trace_length)
    # and computes the pairwise correlation matrix highly efficiently in C.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Ignore warnings temporarily; we handle NaNs manually
        corr_matrix = np.corrcoef(matrix)
    
    # Vectorized fallback for the 'safe_corr' logic (handling zero variance traces)
    # If a row had zero variance, corrcoef outputs NaNs. We replace them with 0.
    np.nan_to_num(corr_matrix, copy=False)
    
    # The diagonal of a correlation matrix must be 1.0
    np.fill_diagonal(corr_matrix, 1.0)
    
    return corr_matrix

def matrix_power_adj(A, p):
    eigvals, eigvecs = np.linalg.eigh(A)

    # Set a threshold for small negative eigenvalues (e.g., 1e-20)
    threshold = 0
    assert np.min(eigvals) > -1e-20, f"Overflow Detected, found negative eigenvalue {np.min(eigvals)}"
    eigvals = np.maximum(eigvals, threshold)  # Replace negative eigenvalues with zero
    # Raise the eigenvalues to the power of p
    eigvals_p = eigvals ** p  # Eigenvalues raised to the power of p
    # Reconstruct the matrix A^p
    A_inv_p = eigvecs @ np.diag(eigvals_p) @ eigvecs.T  # A^p

    return A_inv_p

def clip_eigenvalues(matrix, threshold1=1e5, threshold2=1e-5):
    """
    Clip the eigenvalues of a matrix to a specified threshold.

    Parameters:
    - matrix: 2D numpy array, the matrix whose eigenvalues are to be clipped.
    - threshold: The maximum value to clip eigenvalues to (default is 1e6).

    Returns:
    - clipped_matrix: 2D numpy array, the matrix with clipped eigenvalues.
    """
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Clip eigenvalues that are above the threshold
    clipped_eigenvalues = np.clip(eigenvalues, threshold2, threshold1)

    # Reconstruct the matrix with clipped eigenvalues
    clipped_matrix = eigenvectors @ np.diag(clipped_eigenvalues) @ np.linalg.inv(eigenvectors)

    return clipped_matrix

def fixed_geodes_eff(A, B, p):
    """
    Computes the point t along the geodesic stretching from A to B.

    Parameters:
        A (ndarray): PSD matrix of rank `dim`.
        B (ndarray): PSD matrix of rank `dim`.
        p (float): Desired point along the geodesic (t > 0).
    Returns:
        ndarray: The point t along the geodesic.
    """
    # Compute the ranks of A and B based on non-zero eigenvalues
    rank_A = np.linalg.matrix_rank(A)
    rank_B = np.linalg.matrix_rank(B)
    dim = min(rank_A, rank_B)  # Set dim to the minimum rank
    if dim == np.shape(B)[0] and dim == np.shape(A)[0]:
        return clip_eigenvalues(np.real(filter(A, B, p)))
    # Get the largest `dim` eigenvalues and eigenvectors for A and B
    S1, U1 = np.linalg.eig(A)
    S2, U2 = np.linalg.eig(B)
    S1 = np.real(S1)  # Take only real parts of eigenvalues
    S2 = np.real(S2)

    U1 = U1[:, np.argsort(-S1)[:dim]]
    U2 = U2[:, np.argsort(-S2)[:dim]]
    S1 = -np.sort(-S1)[:dim]
    S2 = -np.sort(-S2)[:dim]

    # Extract eigenvector subspaces
    VA = np.real(U1[:, :dim])  # Ensure eigenvectors are real
    VB = np.real(U2[:, :dim])

    # Singular Value Decomposition (SVD)
    OA, SAB, OB = svd(VA.T @ VB)
    SAB = np.real(SAB)  # Ensure singular values are real

    UA = VA @ OA
    UB = VB @ OB.T
    theta = np.arccos(np.clip(SAB, -1, 1))  # Clip to avoid numerical errors

    # Compute intermediate matrices
    tmp = UB @ pinv(np.diag(np.sin(theta)))
    X = (eye(A.shape[0]).toarray() @ tmp - UA @ UA.T @ tmp)
    U = UA @ np.diag(np.cos(theta * p)) + X @ np.diag(np.sin(theta * p))

    # Compute R2
    RB2 = OB @ np.diag(S2) @ OB.T
    assert np.all(S1 > 0), "Not all eigenvalues are positive!"
    RA = OA.T @ np.diag(np.sqrt(S1)) @ OA
    RAm1 = OA.T @ np.diag(1 / np.sqrt(S1)) @ OA
    eigenvalues = np.real(np.linalg.eigvals(RAm1 @ RB2 @ RAm1))  # Ensure eigenvalues are real
    assert np.all(eigenvalues >= 0), "Not all eigenvalues are positive!"
    R2 = RA @ expm(p * logm(RAm1 @ RB2 @ RAm1)) @ RA
    # Compute the result
    S = U @ R2 @ U.T
    return clip_eigenvalues(np.real(S))  # Ensure final result is real




















