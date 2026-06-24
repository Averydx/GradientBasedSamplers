"""Implementation of the Metropolis Algorithm with Adaptation."""

import jax
import jax.numpy as jnp

from utilities.helpers import cov_update

def mcmc_step(theta_prev, logp_prev, f, key, cov_matrix):
    """Performs a single step of MCMC"""

    prop_key, key = jax.random.split(key)
    theta_prop = jax.random.multivariate_normal(
        prop_key, theta_prev, (2.38**2 / theta_prev.shape[0]) * cov_matrix
    )

    ll_key, key = jax.random.split(key)
    LL_new = f(theta_prop, ll_key)
    LL_old = logp_prev
    alpha = LL_new - LL_old

    key, accept_key = jax.random.split(key)
    accept = jnp.log(jax.random.uniform(minval=0, maxval=1, key=accept_key)) < alpha

    new_theta = jnp.where(accept, theta_prop, theta_prev)
    new_logp = jnp.where(accept, LL_new, LL_old)

    return new_theta, new_logp


def multi_chain_mcmc(f, M, burnin, theta0, adaptive, key, cov_matrix, num_chains):
    """

    Parallelizes the MCMC sampler across multiple chains.

    Parameters :
        f :
            The log-density and its gradient.
        M :
            The number of post-adaptation iterations.
        burnin :
            The number of adaptation iterations.
        theta0 :
            The initial state of the chain.
        adaptive :
            Boolean flag to enable covariance estimation.
        key :
            The jax random key to use in simulation.
        cov_matrix :
            The base covariance matrix to use in sampling.
        num_chains :
            The number of parallel chains to use in sampling.

    Returns :
        The samples and log densities.
    """

    keys = jax.random.split(key, num_chains)

    return jax.vmap(lambda t, k: mcmc(f, M, burnin, t, adaptive, k, cov_matrix))(
        theta0, keys
    )


def mcmc(
    f,
    M,
    burnin,
    theta0,
    adaptive,
    key,
    cov_matrix,
):
    """
    MCMC Metropolis-Hastings sampler.

    Parameters :
        f :
            The log-density.
        M :
            The number of post-adaptation iterations.
        burnin :
            The number of burnin iterations.
        theta0 :
            The initial state of the chain.
        adaptive :
            Boolean flag to enable covariance estimation.
        key :
            The jax random key to use in simulation.
        cov_matrix :
            The base covariance matrix to use in sampling.

    Returns :
        The samples and log densities.
    """

    def one_step(state, m):
        current_theta, current_logp, current_cov, current_mu, current_key = state
        step_key, next_key = jax.random.split(current_key)

        theta_new, logp_new = mcmc_step(
            current_theta, current_logp, f, key=step_key, cov_matrix=current_cov
        )

        def do_update(args):
            c_mu, c_cov, t_new, iter = args
            return cov_update(c_cov, c_mu, t_new, iter, burnin)

        def no_update(args):
            c_mu, c_cov, _, _ = args
            return c_mu, c_cov

        next_mu, next_cov = jax.lax.cond(
            adaptive & (m > burnin),
            do_update,
            no_update,
            (current_mu, current_cov, theta_new, m),
        )

        return (theta_new, logp_new, next_cov, next_mu, next_key), (theta_new, logp_new)

    ll_init_key, key = jax.random.split(key)
    _, (samples, logps) = jax.lax.scan(
        one_step,
        (theta0, f(theta0, ll_init_key), cov_matrix, jnp.zeros_like(theta0), key),
        jnp.arange(0, M + burnin),
    )

    return samples[burnin:, :], logps[burnin:]
