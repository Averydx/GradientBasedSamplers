import jax
import jax.numpy as jnp

from leapfrog import leapfrog
from helpers import find_reasonable_epsilon
from functools import partial


@jax.jit
def kinetic_energy(r, mass_matrix):
    return 1 / 2 * r @ jnp.linalg.pinv(mass_matrix) @ r.T


@partial(jax.jit, static_argnums=(1,))
def hmc_step(thetam, f, *, key, mass_matrix, L, epsilon):
    momenta_key, key = jax.random.split(key)

    # Resample momenta
    r0 = jax.random.multivariate_normal(
        momenta_key, mean=jnp.zeros(len(thetam)), cov=mass_matrix
    )

    thetahat = thetam
    rhat = r0

    logtheta_hat0, gradtheta_hat0 = f(thetahat)

    def step(x,carry):
        thetahat, rhat, gradtheta_hat, _ = carry
        new_carry = leapfrog(thetahat, rhat, gradtheta_hat, epsilon, f)
        return new_carry

    thetahat, rhat, _, _ = jax.lax.fori_loop(0,L,
        step, (thetahat, rhat, gradtheta_hat0, logtheta_hat0)
    )

    LL_old = f(thetam)[0]
    LL_new = f(thetahat)[0]
    H_prop = kinetic_energy(rhat, mass_matrix) - LL_new
    H_curr = kinetic_energy(r0, mass_matrix) - LL_old

    alpha = jnp.minimum(1.0, jnp.exp(-H_prop + H_curr))

    key, accept_key = jax.random.split(key)
    accept = jax.random.uniform(minval=0, maxval=1, key=accept_key) < alpha

    new_theta = jnp.where(accept, thetahat, thetam)
    new_logp = jnp.where(accept, LL_new, LL_old)

    return new_theta, new_logp


def hmc(
    f,
    M,
    Madapt,
    theta0,
    *,
    num_chains=1,
    key=jax.random.key(0),
    mass_matrix=None,
    epsilon=None,
    L=10,
):

    D = len(theta0)
    samples = jnp.zeros((num_chains, M + Madapt, D))
    logps = jnp.zeros((num_chains, M + Madapt))

    logp0, grad0 = f(theta0)
    logps = logps.at[0, 0].set(logp0)
    samples = samples.at[0, 0, :].set(theta0)

    if epsilon is None:
        eps_key, key = jax.random.split(key)
        epsilon = find_reasonable_epsilon(theta0, grad0, logp0, f, eps_key)

    if mass_matrix is None:
        mass_matrix = jnp.eye(D)


    for m in range(1, M + Madapt):
        print(f"Iteration: {m}", end="\r")
        step_key, key = jax.random.split(key)

        theta_new, logp_new = hmc_step(
            samples[0, m - 1, :],
            f,
            key=step_key,
            mass_matrix=mass_matrix,
            L=L,
            epsilon=epsilon,
        )
        samples = samples.at[0, m, :].set(theta_new)
        logps = logps.at[0,m].set(logp_new)

    return samples[:, Madapt:, :], logps[:, Madapt:]
