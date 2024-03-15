"""
BIG TODOS:
    - build data visualization dashboard (dash, plotly)
    - build data pipeline (torch, scikit-learn)
        - preprocessing (cleaning, formatting)
        - transforms (power, differencing, standardization, normalization)
    - add distributed training (ray + aws)
    - add experiment monitoring (w&b)
"""

import os

import ray
import torch
from ray import tune, train
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from src.visualization import build_dashboard
from src.Trainable import Trainable
from src.models import BasicLSTM


# Constants
n_features = 5

# Choose model manually
model = BasicLSTM

config = {

    # Model
    'n_features': n_features,
    'horizon_size': 24,
    'model': model,
    **model.get_hyperspace(),

    # Data
    'log_transform': tune.choice([False, True]),

    # Training
    'train_val_split': 0.8,
    'n_epochs': tune.randint(1, 5),
    'batch_size': tune.randint(64, 256),
    'lr': tune.uniform(0.0001, 0.01),
}


for i in range(n_features):
    config[f'feature_{i}_normalizer'] = tune.choice([MinMaxScaler, StandardScaler, RobustScaler])


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ray.init(local_mode=True, num_cpus=os.cpu_count())

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
        Trainable,
        run_config=train.RunConfig(
            name="bohb_test", stop={"training_iteration": max_iterations}
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

