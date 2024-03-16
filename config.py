import torch.optim
from ray import tune
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from src.models import BasicLSTM


# Runtime Constants
n_features = 5
model = BasicLSTM

# Run config
config = {

    # Settings should define runtime constants
    # Spaces should define ray tunable variables

    'settings': {
        'model': model,
        'n_features': n_features,  # TODO: Size dynamically according to data shape
        'train_val_split': 0.8,
    },

    'model_space': {
        **model.get_hyperspace(),
    },

    'data_space': {
        'log_transform': tune.choice([False, True]),
    },

    'training_space': {
        'n_epochs': tune.randint(1, 5),
        'batch_size': tune.randint(64, 256),
        'optimizer': tune.choice([torch.optim.SGD, torch.optim.Adam]),
        'optimizer_space': {
            'lr': tune.uniform(0.0001, 0.01),
        },
    }
}

for i in range(n_features):
    config['data_space'][f'normalizer_{i}'] = tune.choice([MinMaxScaler, StandardScaler, RobustScaler])