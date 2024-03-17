import os

import ray
import torch
from ray import tune, train
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB

from config import config
from src.visualization import build_dashboard
from src.DataFactory import DataFactory
from src.Trainable import Trainable


if __name__ == "__main__":

    # build_dashboard(DataFactory().pd_data)

    ray.init(local_mode=True)

    max_iterations = 10
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=max_iterations,
        reduction_factor=2,
        stop_last_trials=False,
    )

    bohb_search = TuneBOHB()
    bohb_search = tune.search.ConcurrencyLimiter(bohb_search, max_concurrent=8)

    tuner = tune.Tuner(
        # tune.with_resources(Trainable, resources=tune.PlacementGroupFactory([{'cpu': n_cpus, 'gpu': n_gpus}])),
        tune.with_resources(Trainable, resources={'cpu': 1, 'gpu': 1/8}),
        run_config=train.RunConfig(
            name="bohb_test",
            stop={"training_iteration": max_iterations}
        ),
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="min",
            scheduler=bohb_hyperband,
            search_alg=bohb_search,
            num_samples=16,
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)

