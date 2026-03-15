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
from algorithms.mcmc import multi_chain_mcmc
from utilities.test_distributions import neals_funnel

log_prob = neals_funnel

D = 20
M = 10_000
Madapt = 100
num_chains = 10
key = jax.random.key(0)
init_key, key = jax.random.split(key)

theta0 = jax.random.normal(init_key, (num_chains, D))

mcmc_key, key = jax.random.split(key)
t0 = perf_counter()
samples, lnprob = multi_chain_mcmc(
    log_prob,
    M,
    Madapt,
    theta0,
    key=mcmc_key,
    adaptive=True,
    cov_matrix=jnp.eye(D),
    num_chains=num_chains,
)
t1 = perf_counter()
print(f"Runtime: {t1 - t0} seconds")
print(f"Effective sample size: {effective_sample_size(samples)}")
print(f"Integrated autocorrelation: {autocorr_new(samples)}")
plt.scatter(samples[:, :, 0], samples[:, :, 1],s=0.1)
plt.show()
