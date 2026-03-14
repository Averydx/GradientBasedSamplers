import jax
import jax.numpy as jnp

from utilities.helpers import build_tree, stop_criterion, find_reasonable_epsilon


def nuts(
    f,
    M,
    Madapt,
    theta0,
    *,
    num_chains=1,
    delta=0.6,
    max_depth=5,
    key=jax.random.key(0),
    epsilon=None,
    dual_averaging=False,
):
    """
    The No U Turn Sampler. This is algorithm 6 in Hoffman and Gelman's original paper, 
    utilizing the uniform sampling from C on the fly as well as the dual averaging algorithm.

    Parameters
    ----------
    f : 
        The log density, a function returning both the density and its gradient at the specified point. 
    M : 
        The number of sampling steps to perform after adaptation. 
    Madapt : 
        The number of adaptation steps to perform. If dual_averaging=True 
        this is the period within which the dual averaging algorithm will run. 
    theta0 : 
        Initializer for theta. 
    num_chains : 
        The number of markov chains to run. Defaults to 1. 
    delta : 
        The target acceptance rate used in dual averaging. Defaults to 0.6. 
    max_depth : 
        The maximum tree depth before termination. 
        Number of steps taken before termination on the order of 2^max_depth. 
    key : 
        JAX random key used in algorithm. 
    epsilon : 
        The step size initializer, if dual averaging is false this is the global step size. 
    dual_averaging : 
        Whether to use dual averaging. Defaults to false. 
    """

    D = len(theta0)
    samples = jnp.zeros((num_chains, M + Madapt, D), dtype=jnp.float32)
    lnprob = jnp.zeros((num_chains, M + Madapt), dtype=jnp.float32)

    # Initial step
    logp, grad = f(theta0)
    samples = samples.at[0, 0, :].set(theta0)
    lnprob = lnprob.at[0, 0].set(logp)

    init_epsilon_key, key = jax.random.split(key)

    # Initialize dual averaging algorithm

    if epsilon is None:
        epsilon = find_reasonable_epsilon(theta0, grad, logp, f, init_epsilon_key)

    # Parameters for dual averaging
    gamma = 0.2
    t0 = 10
    kappa = 0.75
    mu = jnp.log(10.0 * epsilon)

    epsilonbar = 1.0 if dual_averaging else epsilon
    Hbar = 1.0

    for m in range(1, M + Madapt):
        # print(f"Iteration: {m} w/ epsilon {epsilon}",end = '\r')
        momenta_key, key = jax.random.split(key)
        # Resample momenta
        r0 = jax.random.normal(momenta_key, D)

        # joint lnp of theta and momentum r
        joint = logp - 0.5 * jnp.dot(r0, r0.T)

        # Resample u
        u_key, key = jax.random.split(key)
        logu = float(joint - jax.random.exponential(u_key))

        # If all steps rejected fall back to previous sample
        samples = samples.at[0, m, :].set(samples[0, m - 1, :])
        lnprob = lnprob.at[0, m].set(lnprob[0, m - 1])

        # initialize the tree
        thetaminus = samples[0, m - 1, :]
        thetaplus = samples[0, m - 1, :]

        rminus = r0
        rplus = r0

        gradminus = grad
        gradplus = grad

        j = 0  # Tree height
        n = 1  # Number of valid points
        s = 1  # Condition for main loop, willkeep going until s == 0

        while s == 1 and j < max_depth:
            # Choose a direction
            direction_key, key = jax.random.split(key)
            v = jax.random.choice(direction_key, a=jnp.array([-1.0, 1.0]))

            subkey, key = jax.random.split(key)
            # Double the size of the tree
            if v == -1:
                (
                    thetaminus,
                    rminus,
                    gradminus,
                    _,
                    _,
                    _,
                    thetaprime,
                    gradprime,
                    logpprime,
                    nprime,
                    sprime,
                    alpha,
                    nalpha,
                ) = build_tree(
                    thetaminus, rminus, gradminus, logu, v, j, epsilon, f, joint, subkey
                )

            else:
                (
                    _,
                    _,
                    _,
                    thetaplus,
                    rplus,
                    gradplus,
                    thetaprime,
                    gradprime,
                    logpprime,
                    nprime,
                    sprime,
                    alpha,
                    nalpha,
                ) = build_tree(
                    thetaplus, rplus, gradplus, logu, v, j, epsilon, f, joint, subkey
                )

            # Use metropolis-hastings to decide whether or not to move to a point from the half-tree we just generated
            metropolis_key, key = jax.random.split(key)
            _tmp = min(1.0, float(nprime) / float(n))
            if (sprime == 1) and (jax.random.uniform(metropolis_key) < _tmp):
                samples = samples.at[0, m, :].set(thetaprime)
                lnprob = lnprob.at[0, m].set(logpprime)
                grad = gradprime

            n += nprime
            # Decide if it's time to stop.
            s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus)
            # Increment depth.
            j += 1

        alpha_avg = alpha / nalpha
        eta = 1 / (m + t0)
        if (m <= Madapt) and dual_averaging:
            Hbar = (1.0 - eta) * Hbar + eta * (delta - alpha_avg)
            epsilon = jnp.exp(mu - jnp.sqrt(m) / gamma * Hbar)
            epsilonbar = jnp.exp(
                (m**-kappa) * jnp.log(epsilon) + (1 - m**-kappa) * jnp.log(epsilonbar)
            )
        else:
            epsilon = epsilonbar

        print(f"Iteration: {m}, tree depth: {j}", end="\r")

    samples = samples[:, Madapt:, :]
    lnprob = lnprob[:, Madapt:]

    return samples, lnprob
