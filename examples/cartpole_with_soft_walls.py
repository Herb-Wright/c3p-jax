"""
uv pip install matplotlib scipy tqdm
uv pip install https://github.com/AndyLamperski/lemkelcp.git
"""
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scipy
import lemkelcp as lcp
from tqdm import trange

from c3p_jax import (
    LCSMatrices, create_rho_schedule, C3Problem, c3p_jit, C3Solution
)



def make_cartpole_with_soft_walls_dynamics(
    ks: float = 100, 
    d: float = 0.35, 
    len_com: float = 1.0,
    mp: float = 0.411,
    dt: float = 0.01,
    mc: float = 0.978,
    len_p: float = 0.6
) -> LCSMatrices:
    g = 9.81
    d1 = d
    d2 = -d
    A = jnp.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, g * mp / mc, 0, 0],
            [0, g * (mc + mp) / (len_com * mc), 0, 0],
        ]
    )
    A = jnp.eye(A.shape[0]) + dt * A
    B = dt * jnp.array([[0], [0], [1 / mc], [1 / (len_com * mc)]])
    D = dt * jnp.array(
        [
            [0, 0],
            [0, 0],
            [(-1 / mc) + (len_p / (mc * len_com)), (1 / mc) - (len_p / (mc * len_com))],
            [
                (-1 / (mc * len_com))
                + (len_p * (mc + mp)) / (mc * mp * len_com * len_com),
                -(
                    (-1 / (mc * len_com))
                    + (len_p * (mc + mp)) / (mc * mp * len_com * len_com)
                ),
            ],
        ]
    )
    E = jnp.array([[-1, len_p, 0, 0], [1, -len_p, 0, 0]])
    F = (1.0 / ks) * jnp.eye(2)
    c = jnp.array([d1, -d2])
    d = jnp.zeros(4)
    H = jnp.zeros((2, 1))
    return LCSMatrices(A=A, B=B, D=D, d=d, E=E, F=F, H=H, c=c)

def get_cost_matrices() -> tuple[jnp.ndarray, jnp.ndarray]:
    Q = jnp.diag(jnp.array([10.0, 2.0, 1.0, 1.0]))
    R = jnp.array([[1.0]])
    return Q, R


def lcs_step(
    lcp_matrices: LCSMatrices, xk: jnp.ndarray, uk: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    b = lcp_matrices.E @ xk + lcp_matrices.H @ uk + lcp_matrices.c
    z, _, msg =  lcp.lemkelcp(np.array(lcp_matrices.F), np.array(b))
    forces = jnp.array(z)
    xkp1 = lcp_matrices.A @ xk + lcp_matrices.B @ uk + lcp_matrices.D @ forces + lcp_matrices.d
    return xkp1, forces







if __name__ == "__main__":
    lcs_matrices = make_cartpole_with_soft_walls_dynamics(
        mp=0.1, mc=1.0, len_p=0.5, d=0.15, ks=12.0, dt=0.1
    )
    lcs_matrices_true = make_cartpole_with_soft_walls_dynamics(mp=0.1, mc=1.0, len_p=0.5, d=0.15, ks=12.0, dt=0.01)
    Q, R = get_cost_matrices()

    Qf = jnp.array(scipy.linalg.solve_discrete_are(lcs_matrices.A, lcs_matrices.B, Q, R))

    T = 10
    n_iters = 15
    nx = 4

    c3_problem = C3Problem()
    c3_problem.Q = Q
    c3_problem.R = R
    c3_problem.set_lcs_matrices(lcs_matrices)
    c3_problem.Qf = Qf
    c3_problem.xd = jnp.zeros([T+1, nx])
    c3_problem.x0 = jnp.array([-0.15, 0.1, 0.0, 0.0])
    var_weights = jnp.ones(c3_problem.n_vars(T))
    var_weights= var_weights.at[-2*T*c3_problem.nc():].set(100.0)
    c3_problem.rho = create_rho_schedule(n_iters, var_weights, mult=2, offset=3)


    system_iter = 500
    x = jnp.zeros((nx, system_iter + 1))
    x = x.at[:, 0].set(c3_problem.x0.ravel())

    for i in trange(system_iter):
        c3_problem.x0 = x[:, i]
        if i % 10 == 0:
            sol: C3Solution = c3p_jit(c3_problem, T, n_iters)

            u_opt = sol.u[0]
        x_next, pred_lambda = lcs_step(lcs_matrices_true, x[:, i], u_opt)
        x = x.at[:, i + 1].set(x_next)

    print("================")
    print(x.T)
    print("================")
    
    dt = 0.01  # same as used for lcs
    time_x = np.arange(0, system_iter * dt + dt, dt)
    plt.plot(time_x, x.T)
    plt.ylim(-0.5, 0.5)
    plt.show()


