import jax.numpy as jnp
import jax
from typing import Callable,Tuple
from functools import partial

@partial(jax.jit,static_argnums = (4,))
def leapfrog(
    theta: jnp.array,
    r: jnp.array,
    grad: jnp.array,
    epsilon: jnp.float64,
    f: Callable[[jnp.array], Tuple[jnp.array,jnp.array]],
):
    """Perform a leapfrog step in phase space."""

    # Half step in r
    rprime = r + 0.5 * epsilon * grad

    # theta step with rprime
    thetaprime = theta + epsilon * rprime

    # log p(p_prime) and its gradient
    logpprime, gradprime = f(thetaprime)

    # another half step in r
    rprime = rprime + 0.5 * epsilon * gradprime

    return thetaprime, rprime, gradprime, logpprime

