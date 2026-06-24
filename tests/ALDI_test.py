import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from time import perf_counter

from algorithms.mala import ALDI
from utilities.test_distributions import rosenbrock

log_prob = rosenbrock

num_particles = 100
iterations = 10000
D = 2
step_size = 0.0001
key = jax.random.key(0)
aldi_key, key = jax.random.split(key)

init_key, key = jax.random.split(key)
init_particles = jax.random.multivariate_normal(
    init_key, mean=jnp.zeros(D), cov=0.1 * jnp.eye(D), shape=(num_particles,)
)

t0 = perf_counter()
(final_particles, _), particles = ALDI(
    jax.value_and_grad(log_prob), init_particles, iterations, step_size, aldi_key, D
)

t1 = perf_counter()
print(f"Runtime: {t1 - t0} seconds")
print(f"Mean Estimate: {jnp.mean(final_particles,axis = 0)}")
print(f"Covariance Estimate: {jnp.cov(final_particles.T)}")

plt.scatter(
    final_particles[:,0], final_particles[:, 1], s=0.1,color = 'tab:blue'
)
plt.scatter(
    init_particles[:,0], init_particles[:, 1], s=0.1,color = 'tab:red'
)

plt.show()