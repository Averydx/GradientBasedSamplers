import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from algorithms.smc import ibis

k = 50
key = jax.random.key(0)
num_particles = 100
dim = 2
key, init_key = jax.random.split(key)
particles = jax.random.normal(init_key, shape=(num_particles, dim))
log_weights = jnp.log(1 / num_particles) * jnp.ones(num_particles)

### Data generation
key, datagen_key = jax.random.split(key)
mu = 10.0
std = 1.0
datalen = 100
obs = mu + std * jax.random.normal(datagen_key, shape=(datalen,))


def likelihood(theta, observations, m, key):
    """
    SMC likelihood function updated to bypass JAX dynamic slicing constraints.
    """
    mask = jnp.arange(observations.shape[0]) < m

    all_log_pdfs = jax.scipy.stats.norm.logpdf(
        observations, loc=theta[0], scale=jnp.exp(theta[1])
    )

    return jnp.sum(jnp.where(mask, all_log_pdfs, 0.0))


ibis_key, key = jax.random.split(key)
particles = ibis(likelihood, obs, k, particles, ibis_key)

plt.title("Joint Distribution of $\\mu$ and $\\sigma$")
plt.scatter(particles[:, 0], jnp.exp(particles[:, 1]))
plt.show()
