import jax.numpy as jnp
import jax
from algorithms.mcmc import mcmc_step


def mcmc_kernel(f_t1, k, particles, log_weights, key):
    cov_matrix = jnp.cov(particles.T, aweights=jnp.exp(log_weights)) + 1e-3 * jnp.eye(
        particles.shape[1]
    )

    def mcmc_step_lax(state, iter):
        particles, logf_t1, iter_key = state
        mcmc_key, next_key = jax.random.split(iter_key)
        iter_keys = jax.random.split(mcmc_key, len(particles))
        new_particles, new_logf_t1 = jax.vmap(mcmc_step, in_axes=(0, 0, None, 0, None))(
            particles, logf_t1, f_t1, iter_keys, cov_matrix
        )

        return (new_particles, new_logf_t1, next_key), (
            new_particles,
            new_logf_t1 != logf_t1,
        )

    key, eval_key = jax.random.split(key)
    eval_keys = jax.random.split(eval_key, len(particles))
    logf_t1_prev = jax.vmap(f_t1, in_axes=(0, 0))(particles, eval_keys)
    key, loop_key = jax.random.split(key)
    final_state, (particle_history, accept_history) = jax.lax.scan(
        mcmc_step_lax, (particles, logf_t1_prev, loop_key), xs=jnp.arange(k)
    )
    final_particles, _, _ = final_state

    return final_particles, accept_history


def multinomial_resampling(log_weights, key):
    num_particles = log_weights.shape[0]
    weights = jnp.exp(log_weights - log_weights.max())
    indices = jax.random.choice(
        key,
        jnp.arange(0, num_particles),
        p=(weights / jnp.sum(weights)).flatten(),
        shape=(num_particles,),
    )

    return indices, weights


def systematic_resampling(log_weights, key):

    num_particles = log_weights.shape[0]
    weights = jnp.exp(log_weights - log_weights.max())
    weights /= jnp.sum(weights)
    cdf = jnp.cumsum(weights)
    u = jax.random.uniform(
        key=key,
        shape=(),
        dtype=log_weights.dtype,
        minval=0,
        maxval=1 / num_particles,
    )

    pointers = u + jnp.arange(num_particles) / num_particles

    indices = jnp.searchsorted(cdf, pointers, side="left")

    return indices, weights


def ibis(f, observations, k, particles0, key):
    """
    Sequential Monte Carlo Sampling using ibis. Implementation follows Chopin's 2020 book.
    """

    num_particles = len(particles0)

    def one_step(state, m):

        prev_particles, prev_log_weights, prev_key = state
        step_key, next_key = jax.random.split(prev_key)

        f_t1 = lambda theta, t1_key: f(theta, observations, m, t1_key)  # noqa: E731
        f_t = lambda theta, t_key: f(theta, observations, m + 1, t_key)  # noqa: E731

        def w_resampling(operand):
            particles, log_weights, r_key = operand
            kernel_key, resampling_key = jax.random.split(r_key)

            indices, _ = systematic_resampling(log_weights, resampling_key)
            log_weights = -jnp.log(num_particles) * jnp.ones(num_particles)
            particles = particles[indices]
            particles, accept_history = mcmc_kernel(
                f_t1, k, particles, log_weights, kernel_key
            )

            # avg_acceptance_rate = jnp.mean(jnp.sum(accept_history, axis=0)) / k

            # jax.debug.print(
            #     "average acceptance rate: {accept_val} over {k_val} iterations",
            #     accept_val=avg_acceptance_rate,
            #     k_val=k,
            # )

            return particles, log_weights

        def wo_resampling(operand):
            particles, log_weights, nr_key = operand
            return particles, log_weights

        ESS = 1 / jnp.sum(jnp.exp(prev_log_weights) ** 2)

        # jax.debug.print("Iteration {m_val} with ESS {ESS_val}", m_val=m, ESS_val=ESS)

        branch_key, step_key = jax.random.split(step_key)
        particles, log_weights = jax.lax.cond(
            ESS < num_particles / 2,
            w_resampling,
            wo_resampling,
            (prev_particles, prev_log_weights, branch_key),
        )

        t_key, _ = jax.random.split(step_key)
        t_keys = jax.random.split(t_key, num_particles)
        log_weights = log_weights + (
            jax.vmap(f_t, in_axes=(0, 0))(particles, t_keys)
            - jax.vmap(f_t1, in_axes=(0, 0))(particles, t_keys)
        )

        log_weights = log_weights - jax.scipy.special.logsumexp(log_weights)

        return (particles, log_weights, next_key), particles

    smc_key, key = jax.random.split(key)
    log_weights = -jnp.log(num_particles) * jnp.ones(num_particles)
    (particles, log_weights, _), _ = jax.lax.scan(
        one_step,
        (particles0, log_weights, smc_key),
        xs=jnp.arange(len(observations)),
    )

    final_resample_key, key = jax.random.split(key)
    indices,_ = systematic_resampling(log_weights, final_resample_key)
    particles = particles[indices]

    return particles


def particle_filter(
    model, observations, initial_particles, key, likelihood, theta, t_range
):
    """
    The bootstrap particle filter.

    Parameters
    ----------
    model :
        The state space model to integrate. Has signature model(IC,theta,keys,t_range),
        where IC is broadcast over the leading particle axis. Parameters are the model
        parameters, keys are the jax PRNG keys, and t_range the length
        of time to integrate(spacing between observations).

    observations :
        A matrix of observations the particle filter will fit to.

    initial_particles :
        The initial particle cloud. The leading axis determines the number of particles
        and the trailing axes must match the model dimension.

    key :
        The jax prng key for random number generation.

    likelihood :
        The likelihood model. The signature is likelihood(particles,observation) -> weights.

    theta :
        A fixed vector of model parameters.

    t_range :
        The length of time to integrate between observations.
    """

    num_particles = initial_particles.shape[0]

    def filter_step(state, i):
        prev_particles, prev_key = state

        forecast_key, next_key = jax.random.split(prev_key)
        forecast_keys = jax.random.split(forecast_key, num_particles)
        forecast_particles = jax.vmap(model, in_axes=(0, None, 0, None))(
            prev_particles, theta, forecast_keys, t_range
        )[:, -1, :]

        log_weights = likelihood(forecast_particles, observations[i], theta)

        resampling_key, next_key = jax.random.split(next_key)
        indices, weights = systematic_resampling(log_weights, resampling_key)
        resampled_particles = forecast_particles[indices]

        return (resampled_particles, next_key), (resampled_particles, weights)

    final_particles, traj = jax.lax.scan(
        filter_step, (initial_particles, key), xs=jnp.arange(0, len(observations))
    )

    return (jnp.swapaxes(traj[0], 0, 1), jnp.swapaxes(traj[1], 0, 1))
