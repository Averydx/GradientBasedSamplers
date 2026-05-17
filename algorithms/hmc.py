"""Implementation of Hamiltonian Monte Carlo"""

import jax
import jax.numpy as jnp

from utilities.integrators import leapfrog
from utilities.helpers import find_reasonable_epsilon,cov_update
from functools import partial


def kinetic_energy(r, mass_matrix):
    """Computes the kinetic energy."""
    return 1 / 2 * r @ jnp.linalg.pinv(mass_matrix) @ r.T


def hmc_step(thetam, f, *, key, mass_matrix, L, epsilon):
    """Performs a single step of hmc."""
    momenta_key, key = jax.random.split(key)

    # Resample momenta
    r0 = jax.random.multivariate_normal(
        momenta_key, mean=jnp.zeros(len(thetam)), cov=mass_matrix
    )

    thetahat = thetam
    rhat = r0

    logtheta_hat0, gradtheta_hat0 = f(thetahat)

    def step(x, carry):
        thetahat, rhat, gradtheta_hat, _ = carry
        new_carry = leapfrog(thetahat, rhat, gradtheta_hat, epsilon, f)
        return new_carry

    thetahat, rhat, _, _ = jax.lax.fori_loop(
        0, L, step, (thetahat, rhat, gradtheta_hat0, logtheta_hat0)
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


def multi_chain_hmc(f, M, Madapt, theta0, key, mass_matrix, epsilon, L, num_chains):
    """
    Parallelizes the HMC sampler across multiple chains. 

    Parameters : 
        f : 
            The log-density and its gradient. 
        M : 
            The number of post-adaptation iterations. 
        Madapt : 
            The number of adaptation iterations. 
        theta0 : 
            The initial state of the chain. 
        key : 
            The jax random key to use in simulation.
        mass_matrix : 
            The mass matrix to use in sampling. 
        epsilon : 
            The integration step size. 
        L : 
            The number of leapfrog steps. 
        num_chains : 
            The number of parallel chains to run. 

    Returns : 
        The samples and log densities. 
    """

    keys = jax.random.split(key, num_chains)

    return jax.vmap(lambda t, k: hmc(f, M, Madapt, t, k, mass_matrix, epsilon, L))(
        theta0, keys
    )


def hmc(
    f,
    M,
    Madapt,
    theta0,
    key,
    mass_matrix,
    epsilon,
    L,
):
    """

    HMC sampler. 

    Parameters : 
        f : 
            The log-density and its gradient. 
        M : 
            The number of post-adaptation iterations. 
        Madapt : 
            The number of adaptation iterations. 
        theta0 : 
            The initial state of the chain. 
        key : 
            The jax random key to use in simulation.
        mass_matrix : 
            The mass matrix to use in sampling. 
        epsilon : 
            The integration step size. 
        L : 
            The number of leapfrog steps. 

    Returns : 
        The samples and log densities. 
    """

    def one_step(state, m):
        current_theta,current_mu,current_cov, current_key = state
        step_key, next_key = jax.random.split(current_key)

        theta_new, logp_new = hmc_step(
            current_theta,
            f,
            key=step_key,
            mass_matrix=jnp.linalg.pinv(current_cov),
            L=L,
            epsilon=epsilon,
        )

        def do_update(args):
            c_mu,c_cov, t_new, iter = args
            return cov_update(c_cov, c_mu, t_new, iter, Madapt)
        
        def no_update(args):
            c_mu, c_cov, _, _ = args
            return c_mu,c_cov

        next_mu, next_cov = jax.lax.cond(
            (m > Madapt),
            do_update,
            no_update,
            (current_mu,current_cov,theta_new,m)
        )

        return (theta_new, next_mu,next_cov, next_key), (theta_new, logp_new)

    _, (samples, logps) = jax.lax.scan(
        one_step, (theta0, jnp.zeros_like(theta0),mass_matrix, key), jnp.arange(0, M + Madapt)
    )

    return samples[Madapt:, :], logps[Madapt:]
