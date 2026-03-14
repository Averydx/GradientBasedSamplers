"""Methods for the implementation of Stein Variational Gradient Descent"""

import jax.numpy as jnp
import jax


def compute_bandwidth(particles):
    sq_dist = (
        jnp.sum(particles**2, axis=1, keepdims=True)
        + jnp.sum(particles**2, axis=1, keepdims=True)
        - 2 * jnp.dot(particles, particles.T)
    )
    median_sq = jnp.median(sq_dist)
    h = median_sq / jnp.log(len(particles) + 1)

    return 1 / jnp.maximum(h, 1e-7)


def svgd(f, kernel, num_particles, iterations, step_size, key, dim_theta):
    """
    The Stein Variational Gradient Descent Algorithm

    Parameters:
    ----------

    f :
        The log density.
    kernel :
        A positive-definite kernel which forms the basis for the optimization.

    Returns :
        The particle distribution, a (num_particles,dim_theta) jax array.
    """

    init_key, key = jax.random.split(key)

    particles = jax.random.multivariate_normal(
        init_key, jnp.zeros(dim_theta), jnp.eye(dim_theta), shape=(num_particles,)
    )

    grad_logp = jax.jit(jax.vmap(jax.grad(f), in_axes=(0,)))
    grad_kernel = jax.jit(
        jax.vmap(jax.grad(kernel, argnums=(0,)), in_axes=(0, None, None))
    )
    vmap_kernel = jax.jit(jax.vmap(kernel, in_axes=(0, None, None)))

    def optim_map(x, particles, bandwidth):
        return jnp.mean(
            vmap_kernel(particles, x, bandwidth)[:, jnp.newaxis] * grad_logp(particles)
            + grad_kernel(particles, x, bandwidth)[0],
            axis=0,
        )

    def update(particle, particles, step_size, bandwidth):
        return particle + step_size * optim_map(particle, particles, bandwidth)

    vmap_update = jax.vmap(update, in_axes=(0, None, None, None))

    def step(particles, iteration):
        bandwidth = compute_bandwidth(particles)
        particles = vmap_update(particles, particles, step_size, bandwidth)
        return particles, iteration

    final_particles, _ = jax.lax.scan(step, (particles), xs=jnp.arange(0, iterations))

    return final_particles
