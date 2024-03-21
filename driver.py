import os
import sys
import logging

import ray
from ray import tune, train
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB

from config import config
from src.visualization import build_dashboard
from src.Trainable import Trainable


def set_logging(level) -> None:
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Add file handlers
    debug_handler = logging.FileHandler('logs/debug.log')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)

    info_handler = logging.FileHandler('logs/info.log')
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)

    # Add stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


if __name__ == "__main__":

    # build_dashboard(DataFactory().pd_data)

    set_logging(logging.INFO)

    ray.init(local_mode=True)

    max_iterations = 30
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=max_iterations,
        reduction_factor=3,
        stop_last_trials=False,
    )

    bohb_search = TuneBOHB()
    bohb_search = tune.search.ConcurrencyLimiter(bohb_search, max_concurrent=4)

    tuner = tune.Tuner(
        tune.with_resources(Trainable, resources={'cpu': 2, 'gpu': 1/4}),
        run_config=train.RunConfig(
            name="bohb_test",
            stop={"training_iteration": max_iterations}
        ),
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=bohb_hyperband,
            search_alg=bohb_search,
            num_samples=16,
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)

