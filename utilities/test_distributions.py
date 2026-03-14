import jax.numpy as jnp
import jax

@jax.jit
def bimodal(p):
    return jnp.logaddexp(-0.5 * jnp.sum(p**2), -0.5 * jnp.sum((p - 4.0) ** 2))

@jax.jit
def rosenbrock(x):
    """
    Computes the log-density of an N-dimensional Rosenbrock distribution.
    Target peak is at (1, 1, ..., 1).
    """
    x_i = x[:-1]
    x_next = x[1:]
    
    # The Rosenbrock formula
    term1 = 100.0 * (x_next - x_i**2)**2
    term2 = (1.0 - x_i)**2
    
    return -jnp.sum(term1 + term2)

@jax.jit
def neals_funnel(theta):
    """
    theta[0] is the 'v' parameter (the scale)
    theta[1:] are the 'x' parameters
    """
    v = theta[0]
    x = theta[1:]
    n = x.shape[0]
    
    # Log-density of v ~ N(0, 3^2)
    # Note: Using 3 as standard deviation (variance = 9)
    logp_v = -0.5 * (v**2 / 9.0 + jnp.log(2 * jnp.pi * 9.0))
    
    # Log-density of x_i ~ N(0, exp(v))
    # log_sigma = v/2, so variance = exp(v)
    logp_x = -0.5 * (jnp.sum(x**2) * jnp.exp(-v) + n * v + n * jnp.log(2 * jnp.pi))
    
    return logp_v + logp_x

@jax.jit
def normal(theta): 
    return -1/2 * jnp.dot(theta,theta.T)