import numpy as np

def generate_hierarchical_dataset(
    depth: int = 5,
    num_features: int = 1000,
    flip_prob: float = 0.1,
    seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a synthetic dataset using a hierarchical diffusion process on a binary tree.
    
    This process mimics how features of data might evolve down an evolutionary or 
    hierarchical tree, resulting in structured correlations between leaves.
    
    @param depth: Depth of the binary tree. Number of resulting items (leaves) is 2^depth.
    @param num_features: Dimensionality of the feature vectors at each node.
    @param flip_prob: Probability of flipping the sign of a feature along a tree edge.
    @param seed: Random seed for reproducibility.
    
    @return: A tuple containing:
        - X: Input matrix, orthogonal (identity) of shape (N_leaves, N_leaves).
        - Y: Output matrix of shape (num_features, N_leaves).
        - U: Left singular vectors of the input-output correlation matrix.
        - S: Singular values.
        - V: Right singular vectors (transposed, shape matches (N_leaves, N_leaves)).
    """
    if seed is not None:
        np.random.seed(seed)
        
    num_leaves = 2 ** depth
    
    # Root node: random +1 or -1 vector
    root_vector = np.random.choice([-1, 1], size=num_features)
    current_level = [root_vector]
    
    # Traverse tree level by level
    for _ in range(depth):
        next_level = []
        for parent in current_level:
            # Create two children per parent
            for _ in range(2):
                flips = np.random.choice([-1, 1], size=num_features, p=[flip_prob, 1 - flip_prob])
                child = parent * flips
                next_level.append(child)
        current_level = next_level
        
    # Stack the leaves' features as columns
    Y = np.column_stack(current_level).astype(float)
    
    # X is an orthogonal identity matrix (one-hot encoding of the items)
    X = np.eye(num_leaves, dtype=float)
    
    # Input-output correlation matrix (Sigma^{31})
    # Since X is Identity, Sigma31 = Y * X^T = Y
    Sigma31 = Y
    
    # SVD extraction
    U, S, Vt = np.linalg.svd(Sigma31, full_matrices=False)
    V = Vt.T
    
    return X, Y, U, S, V

if __name__ == "__main__":
    X, Y, U, S, V = generate_hierarchical_dataset(seed=42)
    print(f"Dataset Generated: X {X.shape}, Y {Y.shape}")
