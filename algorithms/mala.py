import jax
import jax.numpy as jnp


def ALDI(f, initial_particles, iterations, step_size, key, dim_theta):
    num_particles = initial_particles.shape[0]

    def step(state, iter_idx):
        particles, prev_key = state

        particles_mean = jnp.mean(particles, axis=0)
        U = (particles - particles_mean).T

        C_pure = (1.0 / num_particles) * jnp.dot(U, U.T)
        matrix_trace = jnp.trace(C_pure)

        C_half = (1.0 / jnp.sqrt(num_particles)) * U

        max_allowed_trace = 50.0  
        scale_factor = jnp.where(
            matrix_trace > max_allowed_trace, 
            max_allowed_trace / matrix_trace, 
            1.0
        )
        effective_dt = step_size * scale_factor

        C = C_pure + jnp.eye(dim_theta) * 1e-5

        noise_key, next_key = jax.random.split(prev_key)
        eps_matrix = jax.random.normal(noise_key, shape=(num_particles, num_particles))

        _, grads = jax.vmap(f)(particles)
        grads = jnp.clip(grads, -50.0, 50.0)

        def update(particle, grad, eps):
            drift_grad = -jnp.dot(C, grad) * effective_dt
            drift_repulsion = (
                ((dim_theta + 1) / num_particles)
                * (particle - particles_mean)
                * effective_dt
            )
            diffusion = jnp.sqrt(2.0 * effective_dt) * jnp.dot(C_half, eps)

            return drift_grad + drift_repulsion + diffusion

        delta = jax.vmap(update, in_axes=(0, 0, 0))(particles, grads, eps_matrix)
        updated_particles = particles + delta

        return (updated_particles, next_key), updated_particles

    return jax.lax.scan(
        step, (initial_particles, key), xs=jnp.arange(0, iterations)
    )