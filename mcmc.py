import jax
import jax.numpy as jnp

from helpers import cov_update
from functools import partial

@partial(jax.jit, static_argnums=(1,))
def mcmc_step(theta_prev, f, key, cov_matrix):
    prop_key, key = jax.random.split(key)
    theta_prop = jax.random.multivariate_normal(prop_key,
        theta_prev, (2.38**2 / len(theta_prev)) * cov_matrix
    )

    LL_new = f(theta_prop)
    LL_old = f(theta_prev)
    alpha = LL_new - LL_old

    key, accept_key = jax.random.split(key)
    accept = jnp.log(jax.random.uniform(minval=0, maxval=1, key=accept_key)) < alpha

    new_theta = jnp.where(accept, theta_prop, theta_prev)
    new_logp = jnp.where(accept,LL_new,LL_old)

    return new_theta, new_logp

def mcmc(
    f,
    M,
    Madapt,
    theta0,
    *,
    adaptive=True,
    num_chains=1,
    key=jax.random.key(0),
    cov_matrix=None,
):

    D = len(theta0)
    samples = jnp.zeros((num_chains, M + Madapt, D))
    logps = jnp.zeros((num_chains, M + Madapt))
    mu = jnp.zeros(D)

    logp0 = f(theta0)
    logps = logps.at[0, 0].set(logp0)
    samples = samples.at[0, 0, :].set(theta0)

    if cov_matrix is None:
        cov_matrix = jnp.eye(D)

    for m in range(M + Madapt):
        print(f"Iteration: {m}", end="\r")
        step_key, key = jax.random.split(key)

        theta_new, logp_new = mcmc_step(
            samples[0, m - 1, :], f, key=step_key, cov_matrix=cov_matrix
        )

        logps = logps.at[0, m].set(logp_new)
        samples = samples.at[0, m, :].set(theta_new)

        if adaptive and (m > Madapt):
            mu, cov_matrix = cov_update(cov_matrix, mu, samples[0, m, :], m, Madapt)

    return samples[:, Madapt:, :], logps[:, Madapt:]
