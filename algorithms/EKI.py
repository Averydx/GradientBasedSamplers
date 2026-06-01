import jax
import jax.numpy as jnp


def EnsembleKalmanInversion(
    initial_ensemble,
    observations,
    observation_covariance,
    data_map,
    time_step,
    key,
    data_localization_kernel=lambda x: x,
):
    """

    Implementation of the Ensemble Kalman Inversion Algorithm.

    Parameters :
        initial_ensemble :
            The initial ensemble of the parameters. The leading axis is the ensemble size.
        observations :
            The observed data to fit against.
        observation_covariance :
            The covariance structure of the observations.
        data_map :
            The map from the parameters to the data. An arbitrary function
            accepting the ensemble and producing an output of the same shape
            as the observations.
        time_step :
            The size of the time step, the algorithm will terminate when the
            algorithmic time is 1.


    Returns :
        The final parameter ensemble.
    """

    ensemble_size = initial_ensemble.shape[0]
    L = jnp.linalg.cholesky(observation_covariance)

    def step(state, iteration):
        ensemble, prev_key = state

        obs_key, next_key = jax.random.split(prev_key)
        obs_keys = jax.random.split(obs_key, ensemble_size)
        predicted_observations = data_map(ensemble, obs_keys)

        ensemble_anomalies = ensemble - jnp.mean(ensemble, axis=0)
        predicted_obs_anomalies = predicted_observations - jnp.mean(
            predicted_observations, axis=0
        )

        cross_cov = (
            1
            / (ensemble_size - 1)
            * jnp.dot(ensemble_anomalies.T, predicted_obs_anomalies)
        )

        predicted_obs_cov = (
            1
            / (ensemble_size - 1)
            * jnp.dot(predicted_obs_anomalies.T, predicted_obs_anomalies)
        )

        predicted_obs_cov = data_localization_kernel(predicted_obs_cov)

        S = jnp.linalg.pinv(observation_covariance + time_step * predicted_obs_cov)

        posterior_ensemble = (
            ensemble
            + time_step
            * jnp.dot(
                jnp.dot(cross_cov, S), (observations - predicted_observations).T
            ).T
        )

        return (posterior_ensemble, next_key), posterior_ensemble

    return jax.lax.scan(
        step, (initial_ensemble, key), xs=jnp.arange(0, 1 + time_step, time_step)
    )


def ALDI(
    initial_ensemble,
    observations,
    observation_covariance,
    data_map,
    time_step,
    key,
    data_localization_kernel=lambda x: x,
):
    """

    Implementation of the Ensemble Kalman Inversion Algorithm.

    Parameters :
        initial_ensemble :
            The initial ensemble of the parameters. The leading axis is the ensemble size.
        observations :
            The observed data to fit against.
        observation_covariance :
            The covariance structure of the observations.
        data_map :
            The map from the parameters to the data. An arbitrary function
            accepting the ensemble and producing an output of the same shape
            as the observations.
        time_step :
            The size of the time step, the algorithm will terminate when the
            algorithmic time is 1.


    Returns :
        The final parameter ensemble.
    """

    ensemble_size = initial_ensemble.shape[0]
    L = jnp.linalg.cholesky(observation_covariance)

    def step(state, iteration):
        ensemble, prev_key = state

        obs_key, next_key = jax.random.split(prev_key)
        obs_keys = jax.random.split(obs_key, ensemble_size)
        predicted_observations = data_map(ensemble, obs_keys)

        ensemble_anomalies = ensemble - jnp.mean(ensemble, axis=0)
        predicted_obs_anomalies = predicted_observations - jnp.mean(
            predicted_observations, axis=0
        )

        def member_update(member_to_update,members): 
            b = -time_step/ensemble_size * jnp.sum(jnp.dot(predicted_obs_anomalies,))

        return (posterior_ensemble, next_key), posterior_ensemble

    return jax.lax.scan(
        step, (initial_ensemble, key), xs=jnp.arange(0, 1 + time_step, time_step)
    )
