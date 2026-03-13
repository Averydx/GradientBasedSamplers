import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from nuts import nuts
from time import perf_counter

def test_nuts6():

    def rosenbrock_log_pdf(x, a=1.0, b=100.0, sigma=1.0):
        """
        Computes the log-density of the Rosenbrock distribution.
        x: array of shape (2,)
        """
        x1, x2 = x[0], x[1]
        term1 = (a - x1)**2
        term2 = b * (x2 - x1**2)**2
        return -(term1 + term2) / (2 * sigma**2)

    func = jax.jit(jax.value_and_grad(rosenbrock_log_pdf))

    D = 2
    M = 50_000
    Madapt = 0
    key = jax.random.key(0)
    init_key,key = jax.random.split(key)

    theta0 = jax.random.normal(init_key, D)
    delta = 0.95

    nuts_key,key = jax.random.split(key)
    t0 = perf_counter()
    samples, lnprob, epsilon = nuts(func, M, Madapt, theta0, delta=delta,key = nuts_key,epsilon = 0.001)
    t1 = perf_counter()
    print(f'Done. Final epsilon = {epsilon}, runtime {t1 - t0} seconds')
  
    plt.subplot(1,3,1)
    plt.plot(samples[:, 0], samples[:, 1], 'r+')

    plt.subplot(1,3,2)
    plt.hist(samples[:,0], bins=50)
    plt.xlabel("x-samples")

    plt.subplot(1,3,3)
    plt.hist(samples[:,1], bins=50)
    plt.xlabel("y-samples")
    plt.show()

if __name__ == "__main__":
    test_nuts6()