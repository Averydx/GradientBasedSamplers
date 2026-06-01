import jax.numpy as jnp
import jax


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
