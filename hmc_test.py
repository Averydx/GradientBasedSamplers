import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from time import perf_counter

from helpers import (
    autocorrelation,
    auto_window,
    autocorr_new,
    effective_sample_size,
)
from hmc import hmc
from test_distributions import neals_funnel

log_prob = neals_funnel

func = jax.jit(jax.value_and_grad(log_prob))

D = 3
M = 5000
Madapt = 1000
key = jax.random.key(0)
init_key, key = jax.random.split(key)

theta0 = jax.random.normal(init_key, D)

hmc_key, key = jax.random.split(key)
t0 = perf_counter()
samples, lnprob = hmc(func, M, Madapt, theta0, key=hmc_key,epsilon = 0.01,L = 10)
t1 = perf_counter()
print(f"Runtime: {t1 - t0} seconds")
print(f"Effective sample size: {effective_sample_size(samples)}")
print(f"Integrated autocorrelation: {autocorr_new(samples)}")
plt.scatter(samples[0,:,0],samples[0,:,1])
plt.show()




