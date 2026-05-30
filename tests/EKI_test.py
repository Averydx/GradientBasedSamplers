from algorithms.EKI import EnsembleKalmanInversion
import jax
import jax.numpy as jnp


def model(par, key):
    dt = 0.01
    t_range = jnp.arange(0, 2 * jnp.pi + dt, dt)
    phi = 0
    amplitude, vert_shift = par
    amplitude = jnp.exp(amplitude)

    return amplitude * jnp.sin(phi + t_range) + vert_shift


rng_key = jax.random.key(10)
observations_key, rng_key = jax.random.split(rng_key)


def map(par, key):
    sincurve = model(par, key)
    return sincurve


rng_key, noise_key = jax.random.split(rng_key)

theta_true = jnp.array([jnp.log(1.0), 1.0])
y = map(theta_true, noise_key)

dim_output = 2
gamma = 0.1 * jnp.eye(len(y))

y = y + jnp.dot(
    jnp.linalg.cholesky(gamma), jax.random.normal(noise_key, shape=(len(y),))
)

num_ensemble_members = 1000

rng_key, init_ensemble_key, init_key = jax.random.split(rng_key, 3)
initial_ensemble = jax.random.normal(
    init_ensemble_key, (num_ensemble_members, dim_output)
)

(final_ensemble,_), _ = EnsembleKalmanInversion(
    initial_ensemble,
    y,
    gamma,
    lambda x, y: jax.vmap(map, in_axes=(0, 0))(x, y),
    0.1,
    init_key,
)

print(jnp.mean(final_ensemble,axis = 0))
