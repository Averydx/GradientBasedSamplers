import jax.numpy as jnp
import jax
from utilities.leapfrog import leapfrog


@jax.jit
def stop_criterion(thetaminus, thetaplus, rminus, rplus) -> bool:
    """
    Compute the stop condition in the NUTS main loop.
    """

    dtheta = thetaplus - thetaminus

    return (jnp.dot(dtheta, rminus) >= 0) & (jnp.dot(dtheta, rplus) >= 0)


def build_tree(theta, r, grad, logu, v, j, epsilon, f, joint0, key):
    """The tree building recursion in NUTS."""

    if j == 0:
        # Base case: single leapfrog step
        thetaprime, rprime, gradprime, logpprime = leapfrog(
            theta, r, grad, v * epsilon, f
        )
        joint = logpprime - 0.5 * jnp.dot(rprime, rprime)
        # Check if the new point is in the slice
        nprime = int(logu < joint)
        # Check if the simulation is very inaccurate
        sprime = int((logu - 1000.0) < joint)
        # Set the return values, minus=plus here
        thetaminus = thetaprime
        thetaplus = thetaprime
        rminus = rprime
        rplus = rprime
        gradminus = gradprime
        gradplus = gradprime
        # Compute the acceptance probability
        alphaprime = min(1.0, jnp.exp(joint - joint0))
        nalphaprime = 1

    else:
        # Recursion
        subkey, key = jax.random.split(key)
        (
            thetaminus,
            rminus,
            gradminus,
            thetaplus,
            rplus,
            gradplus,
            thetaprime,
            gradprime,
            logpprime,
            nprime,
            sprime,
            alphaprime,
            nalphaprime,
        ) = build_tree(theta, r, grad, logu, v, j - 1, epsilon, f, joint0, subkey)
        # No need to keep going if the stopping criteria were met in the first subtree.
        if sprime == 1:
            subkey, key = jax.random.split(key)
            if v == -1:
                (
                    thetaminus,
                    rminus,
                    gradminus,
                    _,
                    _,
                    _,
                    thetaprime2,
                    gradprime2,
                    logpprime2,
                    nprime2,
                    sprime2,
                    alphaprime2,
                    nalphaprime2,
                ) = build_tree(
                    thetaminus,
                    rminus,
                    gradminus,
                    logu,
                    v,
                    j - 1,
                    epsilon,
                    f,
                    joint0,
                    subkey,
                )
            else:
                (
                    _,
                    _,
                    _,
                    thetaplus,
                    rplus,
                    gradplus,
                    thetaprime2,
                    gradprime2,
                    logpprime2,
                    nprime2,
                    sprime2,
                    alphaprime2,
                    nalphaprime2,
                ) = build_tree(
                    thetaplus,
                    rplus,
                    gradplus,
                    logu,
                    v,
                    j - 1,
                    epsilon,
                    f,
                    joint0,
                    subkey,
                )
            # Choose which subtree to propagate a sample up from.
            sample_key, key = jax.random.split(key)
            if jax.random.uniform(sample_key) < (
                float(nprime2) / max(float(int(nprime) + int(nprime2)), 1.0)
            ):
                thetaprime = thetaprime2
                gradprime = gradprime2
                logpprime = logpprime2
            # Update the number of valid points.
            nprime = int(nprime) + int(nprime2)
            # Update the stopping criterion.
            sprime = int(
                sprime
                and sprime2
                and stop_criterion(thetaminus, thetaplus, rminus, rplus)
            )
            # Update the acceptance probability statistics.
            alphaprime = alphaprime + alphaprime2
            nalphaprime = nalphaprime + nalphaprime2

    return (
        thetaminus,
        rminus,
        gradminus,
        thetaplus,
        rplus,
        gradplus,
        thetaprime,
        gradprime,
        logpprime,
        nprime,
        sprime,
        alphaprime,
        nalphaprime,
    )


def find_reasonable_epsilon(theta0, grad0, logp0, f, key):
    """Heuristic for choosing an initial value of epsilon"""
    epsilon = 0.1
    momenta_key, key = jax.random.split(key)
    r0 = jax.random.normal(momenta_key, len(theta0))

    def log_joint(theta, r):
        return f(theta)[0] - 1 / 2 * jnp.dot(r, r)

    # Figure out what direction we should be moving epsilon.
    thetaprime, rprime, _, _ = leapfrog(theta0, r0, grad0, epsilon, f)

    a = (
        2.0 * ((log_joint(thetaprime, rprime) - log_joint(theta0, r0)) > jnp.log(0.5))
        - 1.0
    )

    while (log_joint(thetaprime, rprime) - log_joint(theta0, r0)) ** a > jnp.log(2**-a):
        epsilon = 2**a * epsilon
        thetaprime, rprime, _, _ = leapfrog(theta0, r0, grad0, epsilon, f)

    print("find_reasonable_epsilon=", epsilon)

    return max(0.1, epsilon)


def next_pow_two(n):
    """Find the next power of two < n."""

    def cond_fn(state):
        i, n = state
        return i < n

    def body_fn(state):
        i, n = state
        return jnp.array([i << 1, n])

    return jax.lax.while_loop(cond_fn, body_fn, jnp.array([1, n]))[0]


def autocorrelation(chain):
    """Compute the autocorrelation of a markov chain."""
    num_samples = chain.shape[0]
    n_fft = 2 * next_pow_two(num_samples)

    # Compute the FFT and then auto-correlation function
    f = jnp.fft.rfft(chain - jnp.mean(chain, axis=0), n=n_fft, axis=0)
    acf = jnp.fft.irfft(f * jnp.conjugate(f), n=n_fft, axis=0)
    acf = acf[:num_samples]
    acf /= acf[0]

    return acf


def auto_window(taus, c):
    """Automated windowing procedure following Sokal (1989)"""
    indices = jnp.arange(taus.shape[0])[:, jnp.newaxis]

    m = indices < c * taus

    window_idxs = jnp.argmin(m, axis=0)

    is_always_true = jnp.all(m, axis=0)

    return jnp.where(is_always_true, taus.shape[0] - 1, window_idxs)


def autocorr_new(y, c=5.0):
    """Compute the autocorrelation across markov chains."""
    num_chains, num_samples, num_dims = y.shape

    f = jnp.zeros((num_samples, num_dims))
    for yy in y:
        f += autocorrelation(yy)
    f /= num_chains

    taus = 2.0 * jnp.cumsum(f, axis=0) - 1.0
    window = auto_window(taus, c)

    return jnp.diag(taus[window])


def effective_sample_size(chains):
    """The effective sample size across the chains."""
    num_chains, num_samples, num_dim = chains.shape

    return num_samples / autocorr_new(chains)


def cov_update(cov, mu, theta_val, iteration, burn_in):
    """Adaptive update step, geometric cooling g ensures ergodicity of the markov chain
    as the iteration count goes to infinity.

    cov :
        The running covariance.
    mu :
        The running mean.
    iteration :
        The current iteration.
    burn_in :
        The number of burn_in iterations.
    """

    g = (iteration - burn_in + 1) ** (-0.4)
    mu = (1.0 - g) * mu + g * theta_val
    m_theta = theta_val - mu

    r_cov = (1.0 - g) * cov + g * jnp.outer(m_theta, m_theta)

    return mu, r_cov


def gr_statistic(chains):
    """Computes the Gelman-Rubin statistic for the
    given chains."""

    num_chains, num_samples, num_dim = chains.shape

    chain_mean = jnp.mean(chains, axis=1)
    grand_mean = jnp.mean(chain_mean, axis=0)

    between_chain_var = (
        num_samples
        / (num_chains - 1.0)
        * jnp.sum((chain_mean - grand_mean) ** 2, axis=0)
    )

    within_chain_var = jnp.var(chains, axis=1)

    W = 1 / num_chains * jnp.sum(within_chain_var)

    R = (
        (num_samples - 1.0) / num_samples * W + 1 / num_samples * between_chain_var
    ) / W

    return R
