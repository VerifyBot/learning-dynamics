import numpy as np

def analytical_solution(S: np.ndarray, epochs: int = 1500, lr: float = 0.5e-3, u0: float = 1e-3) -> np.ndarray:
    """
    Computes the analytical solution for mode strengths over time based on Andrew Saxe's exact solution.
    
    The theoretical equation for the learning dynamics of deep linear networks is:
        u_f(t) = s * exp(2 * s * t / tau) / (exp(2 * s * t / tau) - 1 + s / u0)
    
    where `s` is the singular value, and `tau = 1/lr`.
    
    @param S: Array of singular values from the dataset's input-output correlation matrix.
    @param epochs: Number of training epochs to simulate.
    @param lr: Learning rate to mimic.
    @param u0: Initial mode strength scalar.
    
    @return: Analytically derived mode strengths over time, shape (epochs, len(S)).
    """
    t = np.arange(epochs)
    tau = 1.0 / lr
    analytical_curves = []
    
    for s_alpha in S:
        exp_term = np.exp(2 * s_alpha * t / tau)
        uf = (s_alpha * exp_term) / (exp_term - 1 + s_alpha / u0)
        analytical_curves.append(uf)
        
    return np.array(analytical_curves).T
