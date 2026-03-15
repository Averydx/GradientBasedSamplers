import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from time import perf_counter

from utilities.helpers import (
    autocorrelation,
    auto_window,
    autocorr_new,
    effective_sample_size,
)
from algorithms.svgd import svgd
from utilities.test_distributions import rosenbrock

log_prob = rosenbrock

def RBF_Kernel(x,x_prime,bandwidth): 
    sq_dist = jnp.sum((x-x_prime)**2)
    return jnp.exp(-bandwidth * sq_dist)

num_particles = 10000
iterations = 1000
D = 2
step_size = 0.01
key = jax.random.key(0)
svgd_key, key = jax.random.split(key)

t0 = perf_counter()
particles = svgd(log_prob,RBF_Kernel,num_particles,iterations,step_size,svgd_key,D)
t1 = perf_counter()
print(f"Runtime: {t1 - t0} seconds")

plt.scatter(particles[:, 0], particles[:, 1],s=0.1)
plt.show()
