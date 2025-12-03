"""
The purpose of this file is to implement C3+ in JAX
"""
import jax
import jax.numpy as jnp
from chex import dataclass

from .utils import place


@dataclass
class LCSMatrices:
    """
    matrices for the LCS problem defined by

    xₖ₊₁ = A xₖ + B uₖ + D λₖ + d

    0 ≤ λₖ ⟂ (E xₖ + F λₖ + H uₖ + c) ≥ 0
    """
    A: jnp.ndarray  # (nx, nx)
    B: jnp.ndarray  # (nx, nu)
    D: jnp.ndarray  # (nx, nc)
    d: jnp.ndarray  # (nx,)
    E: jnp.ndarray  # (nc, nx)
    F: jnp.ndarray  # (nc, nc)
    H: jnp.ndarray  # (nc, nu)
    c: jnp.ndarray  # (nc,)


@dataclass
class C3Problem:
    """
    Definition of a C3 complementarity problem and ADMM solver parameters
    """
    Q: jnp.ndarray = None
    Qf: jnp.ndarray = None
    R: jnp.ndarray = None
    A: jnp.ndarray = None
    B: jnp.ndarray = None
    D: jnp.ndarray = None
    d: jnp.ndarray = None
    E: jnp.ndarray = None
    F: jnp.ndarray = None
    H: jnp.ndarray = None
    c: jnp.ndarray = None
    rho: jnp.ndarray = None  # (n_iter, n_vars)
    x0: jnp.ndarray = None   # (n_x)
    xd: jnp.ndarray = None  #(T+1, n_x)

    def nx(self) -> int:
        return self.Q.shape[0]
    
    def nu(self) -> int:
        return self.R.shape[0]
    
    def nc(self) -> int:
        return self.F.shape[0]
    
    def n_vars(self, T: int) -> int:
        return (T+1)*self.nx() + T*self.nu() + 2*T*self.nc()
    
    def get_lcs_matrices(self) -> LCSMatrices:
        return LCSMatrices(A=self.A, B=self.B, D=self.D, d=self.d, E=self.E, F=self.F, H=self.H, c=self.c)
    
    def set_lcs_matrices(self, lcs_matrices: LCSMatrices):
        self.A = lcs_matrices.A
        self.B = lcs_matrices.B
        self.D = lcs_matrices.D
        self.d = lcs_matrices.d
        self.E = lcs_matrices.E
        self.F = lcs_matrices.F
        self.H = lcs_matrices.H
        self.c = lcs_matrices.c


@dataclass
class C3Solution:
    """
    The solution to the C3 problem: `x`, `u`, and `forces`
    """
    x: jnp.ndarray  # (T+1, n_x)
    u: jnp.ndarray  # (T, n_u)
    forces: jnp.ndarray  # (T, n_c)
    # costs: jnp.ndarray
    # violations: jnp.ndarray



def c3p(c3_problem: C3Problem, T: int, n_iters: int, end_on_qp: bool = True) -> C3Solution:
    """
    Jittable function that runs the C3+ ADMM algorithm.
    """
    n_vars = (T+1)*c3_problem.nx() + T*c3_problem.nu() + 2*T*c3_problem.nc()
    delta = jnp.zeros(n_vars)
    w = jnp.zeros(n_vars)

    for i in range(n_iters):
        rho = c3_problem.rho[i]
        if i != 0:  # rescale w if rho changes
            w =  c3_problem.rho[i-1] * w / rho

        # z step
        M1, v1, M2, v2 = build_qp_matrices(
            c3_problem.get_lcs_matrices(), c3_problem.Q, c3_problem.R, w, delta, T, rho, 
            c3_problem.x0, c3_problem.xd, c3_problem.Qf
        )
        z = solve_equality_qp(M1, v1, M2, v2)

        # λ step
        delta = project_step(z, w, T, c3_problem.nc())

        # update w
        w = w + z - delta
    
    if end_on_qp:
        M1, v1, M2, v2 = build_qp_matrices(
            c3_problem.get_lcs_matrices(), c3_problem.Q, c3_problem.R, w, delta, T, rho, 
            c3_problem.x0, c3_problem.xd, c3_problem.Qf
        )
        z = solve_equality_qp(M1, v1, M2, v2)
    z = delta

    c3_sol = destructure_vars(
        z, T, c3_problem.nx(), c3_problem.nu(), c3_problem.nc()
    )
    return c3_sol


def c3p_with_costs(
    c3_problem: C3Problem, T: int, n_iters: int, end_on_qp: bool = True
) -> tuple[C3Solution, jnp.ndarray, jnp.ndarray]:
    """
    Jittable function that runs the C3+ ADMM algorithm.
    """
    n_vars = (T+1)*c3_problem.nx() + T*c3_problem.nu() + 2*T*c3_problem.nc()
    delta = jnp.zeros(n_vars)
    w = jnp.zeros(n_vars)
    costs = jnp.zeros(n_iters)
    violations = jnp.zeros(n_iters)

    for i in range(n_iters):
        rho = c3_problem.rho[i]
        if i != 0:  # rescale var if rho changes
            w =  c3_problem.rho[i-1] * w / rho
        M1, v1, M2, v2 = build_qp_matrices(
            c3_problem.get_lcs_matrices(), c3_problem.Q, c3_problem.R, w, delta, T, rho, 
            c3_problem.x0, c3_problem.xd, c3_problem.Qf
        )
        z = solve_equality_qp(M1, v1, M2, v2)
        delta = project_step(z, w, T, c3_problem.nc())
        costs = costs.at[i].set(_eval_cost(delta, c3_problem, T))
        violations = violations.at[i].set(jnp.sum(jnp.abs(M2 @ delta - v2)))
        w = w + z - delta
    
    if end_on_qp:
        M1, v1, M2, v2 = build_qp_matrices(
            c3_problem.get_lcs_matrices(), c3_problem.Q, c3_problem.R, w, delta, T, rho, 
            c3_problem.x0, c3_problem.xd, c3_problem.Qf
        )
        z = solve_equality_qp(M1, v1, M2, v2)

    c3_sol = destructure_vars(
        z, T, c3_problem.nx(), c3_problem.nu(), c3_problem.nc()
    )
    return c3_sol, costs, violations




def _eval_cost(delta: jnp.ndarray, c3_problem: C3Problem, T) -> jnp.ndarray:
    Ntot = delta.shape[0]
    nx = c3_problem.nx()
    Nx = (T+1) * nx
    ox = 0
    ou = ox + Nx
    Q = c3_problem.Q
    Qf = c3_problem.Qf
    R = c3_problem.R
    M1 = jnp.zeros((Ntot, Ntot))

    # Quadratic cost: sum xₖᵀ Q xₖ + uₖᵀ R uₖ 
    # Add Q on each xₖ block
    Qblk = jnp.kron(jnp.eye(T+1), Q)
    Qblk = Qblk.at[-nx:, -nx:].set(Qf)
    M1 = place(M1, Qblk, ox, ox)

    # Add R on each uₖ block
    Rblk = jnp.kron(jnp.eye(T), R)
    M1 = place(M1, Rblk, ou, ou)

    change_d = jnp.zeros(Ntot)
    change_d = change_d.at[:Nx].set(c3_problem.xd.flatten())
    v1 = - 2 * M1 @ change_d

    return delta.T @ M1 @ delta + v1 @ delta + v1.T @ M1 @ v1
    


def project_step(z: jnp.ndarray, w: jnp.ndarray, T: int, n_c: int, mult: float = 1.0) -> jnp.ndarray:
    """solves the C3+ ADMM projection step, returning δ, where λ ⟂ η"""
    vars_not = z + w
    new_delta = jnp.copy(vars_not)
    lambda_not = vars_not[-2*T*n_c:-T*n_c]
    eta_not = vars_not[-T*n_c:]

    new_delta = new_delta.at[-2*T*n_c:-T*n_c].set(lambda_not * (lambda_not >= 0) * (lambda_not >= mult * eta_not))
    new_delta = new_delta.at[-T*n_c:].set(eta_not * (eta_not >= 0) * (mult * eta_not > lambda_not))
    return new_delta


def destructure_vars(z: jnp.ndarray, T: int, n_x: int, n_u: int, n_c: int) -> C3Solution:
    """
    Extract x and u from the flattened decision vector z.

    z is ordered as:
        [ x(0:T+1),  u(0:T),  λ(0:T),  η(0:T) ]

    - x: (T+1)*n_x
    - u: T*n_u
    - λ: T*n_c      
    - η: T*n_c      (ignored)
    """

    Nx = (T + 1) * n_x
    Nu = T * n_u
    Nc = T * n_c

    x_flat = z[:Nx]
    u_flat = z[Nx:Nx + Nu]
    forces_flat = z[Nx + Nu: Nx + Nu + Nc]

    x = x_flat.reshape(T + 1, n_x)     # states 0...T
    u = u_flat.reshape(T, n_u)         # controls 0...T-1
    forces = forces_flat.reshape(T, n_c)

    return C3Solution(x=x, u=u, forces=forces)


def solve_equality_qp(M1, v1, M2, v2):
    """
    Solve:
         min   zᵀ M1 z + v1ᵀ z
         s.t.  M2 z = v2

    via KKT linear system:
    
        [ 2M1   M2ᵀ ] [z     ] = [ -v1 ]
        [ M2     0  ] [λ_mult]   [  v2 ]
    """

    n = M1.shape[0]          # number of primal vars z
    m = M2.shape[0]          # number of equality constraints
    KKT = jnp.block([
        [2 * M1,          M2.T      ],
        [M2,              jnp.zeros((m, m)) ]
    ])
    rhs = jnp.concatenate([-v1, v2])
    sol = jnp.linalg.solve(KKT, rhs)
    z = sol[:n]
    return z


def build_qp_matrices(
    matrices: LCSMatrices,
    Q: jnp.ndarray,
    R: jnp.ndarray,
    w: jnp.ndarray, 
    delta: jnp.ndarray, 
    T: int, 
    rho: float | jnp.ndarray,  
    x0: jnp.ndarray,  # (n_x)
    xd: jnp.ndarray,  # (T+1, n_x)
    Qf: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Build (M1, v1, M2, v2) for the quadratic program:
    
      minimize   zᵀ M1 z + v1ᵀ z
      subject to M2 z = v2

    z = [x; u; λ; η]
    - x: (T+1, nx)
    - u: (T, nu)
    - λ: (T, nc)
    - η: (T, nc)
    """
    # Get the LCSMatrices
    A = matrices.A 
    B = matrices.B 
    D = matrices.D 
    d = matrices.d 
    E = matrices.E 
    F = matrices.F 
    H = matrices.H 
    c = matrices.c 

    nx = Q.shape[0]
    nu = R.shape[0]
    nc = F.shape[0]

    # flattened sizes
    Nx = (T + 1) * nx
    Nu = T * nu
    Nl = T * nc
    Ne = T * nc

    # start positions
    ox = 0
    ou = ox + Nx
    ol = ou + Nu
    oe = ol + Nl
    Ntot = oe + Ne

    # 1) Quadratic Cost Matrix M1
    M1_no_ridge = jnp.zeros((Ntot, Ntot))

    # Quadratic cost: sum xₖ^T Q xₖ + uₖ^T R uₖ 
    # Add Q on each xₖ block
    Qblk = jnp.kron(jnp.eye(T+1), Q)
    Qblk = Qblk.at[-nx:, -nx:].set(Qf)
    M1_no_ridge = place(M1_no_ridge, Qblk, ox, ox)

    # Add R on each uₖ block
    Rblk = jnp.kron(jnp.eye(T), R)
    M1_no_ridge = place(M1_no_ridge, Rblk, ou, ou)

    # Regularization ρ ||z||^2  -> add ρ I
    M1 = M1_no_ridge + jnp.diag(rho)

    # 2) Linear term v1
    v1 = jnp.zeros((Ntot,))
    # add ρ (w - delta)
    v1 = v1 + 2 * rho * (w - delta)

    # 3) Equality constraints M2 z = v2

    # total equalities: T * nx (dynamics) + T * nc (η-def)
    Neq = (T+1) * nx + T * nc
    M2 = jnp.zeros((Neq, Ntot))
    v2 = jnp.zeros((Neq,))

    # Dynamics constraints
    # xₖ₊₁ - A xₖ - B uₖ - D λₖ = d

    M2 = place(M2, jnp.eye(nx), 0, 0)
    v2 = v2.at[0:nx].set(x0)

    row = nx
    for k in range(T):
        # indices
        rx = ox + k * nx
        rx_next = ox + (k + 1) * nx
        ru = ou + k * nu
        rl = ol + k * nc

        # row block indices
        r0 = row
        r1 = row + nx

        # xₖ coefficient: -A
        M2 = place(M2, -A, r0, rx)
        # x_{k+1} coefficient: +I
        M2 = place(M2, jnp.eye(nx), r0, rx_next)
        # uₖ coefficient: -B
        M2 = place(M2, -B, r0, ru)
        # λₖ coefficient: -D
        M2 = place(M2, -D, r0, rl)

        # RHS: d
        v2 = v2.at[r0:r1].set(d)

        row += nx

    # η constraints
    # ηₖ - E xₖ - F λₖ - H uₖ = c
    for k in range(T):
        # indices
        rx = ox + k * nx
        ru = ou + k * nu
        rl = ol + k * nc
        re = oe + k * nc

        r0 = row
        r1 = row + nc

        # ηₖ: +I
        M2 = place(M2, jnp.eye(nc), r0, re)
        # xₖ: -E
        M2 = place(M2, -E, r0, rx)
        # λₖ: -F
        M2 = place(M2, -F, r0, rl)
        # uₖ: -H
        M2 = place(M2, -H, r0, ru)

        v2 = v2.at[r0:r1].set(c)

        row += nc

    # make adjustment for xd  (v := (v - 2 M xd))
    change_d = jnp.zeros(Ntot)
    change_d = change_d.at[:Nx].set(xd.flatten())
    v1 = v1 - 2 * M1_no_ridge @ change_d

    return M1, v1, M2, v2



# Make a jittable wrapper
c3p_jit = jax.jit(c3p, static_argnums=(1, 2, 3))








