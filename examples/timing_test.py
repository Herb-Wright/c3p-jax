import time

from tqdm import trange
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from c3p_jax import LCSMatrices, C3Problem, C3Solution, create_rho_schedule, c3p



def create_rand_lcs_system(
    key: jax.random.PRNGKey, nx: int, nu: int, nc: int, mult = 0.1,
) -> LCSMatrices:
    A = mult * jax.random.normal(key, (nx, nx))
    B = mult * jax.random.normal(key, (nx, nu))
    D = mult * jax.random.normal(key, (nx, nc))
    d = mult * jax.random.normal(key, (nx,))
    E = mult * jax.random.normal(key, (nc, nx))
    F = mult * jax.random.normal(key, (nc, nc))
    H = mult * jax.random.normal(key, (nc, nu))
    c = mult * jax.random.normal(key, (nc,))
    return LCSMatrices(A=A, B=B, D=D, d=d, E=E, F=F, H=H, c=c)


def create_c3_problem_from_lcs(key: jax.random.PRNGKey, lcs: LCSMatrices, nx: int, nu: int, T: int, n_iters: int) -> C3Problem:
    c3_problem = C3Problem()
    # c3_problem.Q = jnp.eye(nx)
    c3_problem.Q = jax.random.normal(key, (nx, nx))
    c3_problem.Q = c3_problem.Q.T @ c3_problem.Q
    # c3_problem.R = jnp.eye(nu)
    c3_problem.R = jax.random.normal(key, (nu, nu))
    c3_problem.R = c3_problem.R.T @ c3_problem.R
    c3_problem.set_lcs_matrices(lcs)
    c3_problem.Qf = jnp.eye(nx)
    c3_problem.xd = jnp.zeros([T+1, nx])
    c3_problem.x0 = 0.1 * jnp.ones(nx)
    var_weights = jnp.ones(c3_problem.n_vars(T))
    var_weights= var_weights.at[-2*T*c3_problem.nc():].set(100.0)
    c3_problem.rho = create_rho_schedule(n_iters, var_weights, mult=2, offset=3)
    return c3_problem



if __name__ == "__main__":
    nx = 58
    nu = 3
    nc = 76
    T = 7
    n_iter = 3

    get_c3_problem_vmap = jax.jit(jax.vmap(
        lambda k: create_c3_problem_from_lcs(k, create_rand_lcs_system(k, nx, nu, nc), nx, nu, T, n_iter)
    ))

    c3p_vmap_jit = jax.jit(jax.vmap(lambda p: c3p(p, T, n_iter)))

    nsamples = [1, 3, 5,  10, 30, 100, 300, 500]

    times_all = []
    for ns in nsamples:
        print(f"starting with ns: {ns}")
        root_key = jax.random.PRNGKey(0)
        batch_keys = jax.random.split(root_key, ns)

        # warm up
        c3_problem = get_c3_problem_vmap(batch_keys)
        for i in range(10):
            c3p_vmap_jit(c3_problem)

        print("all warm!")

        times = []
        for i in trange(30):
            c3_problem: C3Problem = get_c3_problem_vmap(batch_keys)

            start = time.time()
            sol = c3p_vmap_jit(c3_problem)
            jax.block_until_ready(sol)
            end = time.time()

            times.append(end - start)
        print(f"ns {ns} had avg time: {sum(times) / 30}")
        times_all.append(times)
    

    means = []
    errors = []

    for group in times_all:
        mean = np.mean(group)
        means.append(mean)
        
        n = len(group)
        sem = np.std(group, ddof=1) / np.sqrt(n)
        
        t_critical = stats.t.ppf(0.975, df=n-1)
        margin_of_error = t_critical * sem
        
        errors.append(margin_of_error)

    plt.figure(figsize=(6, 3))

    print("means", means)

    plt.errorbar(nsamples, means, yerr=errors, fmt='-o', capsize=5, color='blue', ecolor='red')

    plt.xlabel('$n$ samples')
    plt.ylabel('time (s)')
    plt.xticks(nsamples)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')

    plt.tight_layout()
    # plt.savefig("time.png", dpi=200)
    plt.show()