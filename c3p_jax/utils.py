import jax.numpy as jnp


def place(M, block, r, c):
    """Place a block into big matrix M at row r, col c."""
    rr, cc = block.shape
    M = M.at[r:r+rr, c:c+cc].set(block)
    return M

def create_rho_schedule(
    n_iter: int, 
    var_weights: jnp.ndarray, 
    mult: float = 8, 
    offset: int = 5, 
    start_with_0_rho: bool = False,
) -> jnp.ndarray:
    rho = jnp.outer(mult ** (jnp.arange(n_iter, dtype=var_weights.dtype) - offset), var_weights)
    if start_with_0_rho:
        rho = rho.at[0].set(1e-7)
    return rho