import pandas as pd
import torch
from copy import copy


class DataFactory:

    def __init__(self):
        self.pd_data = pd.read_excel("data/energy.xlsx")

        # TODO: Clean data (NaNs)
        # TODO: Interpolate missing data points
        self.np_data = self.pd_data.iloc[:, 2:].to_numpy()
        self.data = torch.Tensor(self.np_data)

        self.normalizers = []

    def generate_datasets(self, config):

        out_data = copy(self.data)

        # Power transforms
        if config['log_transform']:
            out_data = torch.log(out_data)

        # Split train / val
        x_train = out_data[:int(config['train_val_split']*out_data.size(0))]
        x_val = out_data[int(config['train_val_split']*out_data.size(0)):]

        # Difference transforms

        # Feature independent normalization
        for i in range(out_data.size(1)):
            normalizer = config[f'feature_{i}_normalizer']()
            np_norm_feat_train = normalizer.fit_transform(torch.unsqueeze(x_train[:, i], dim=1))
            np_norm_feat_val = normalizer.transform(torch.unsqueeze(x_val[:, i], dim=1))
            x_train[:, i] = torch.squeeze(torch.Tensor(np_norm_feat_train), dim=1)
            x_val[:, i] = torch.squeeze(torch.Tensor(np_norm_feat_val), dim=1)

            self.normalizers.append(normalizer)

        n_train_sequences = x_train.size(0) - config['seq_length']
        n_val_sequences = x_val.size(0) - config['seq_length']

        # Build training sequences and targets
        train_sequences = torch.empty(size=(n_train_sequences, config['seq_length'], x_train.size(1)))
        train_targets = torch.empty(size=(n_train_sequences, config['horizon_size']))
        val_sequences = torch.empty(size=(n_val_sequences, config['seq_length'], x_val.size(1)))
        val_targets = torch.empty(size=(n_val_sequences, config['horizon_size']))

        for i in range(n_train_sequences-config['horizon_size']):
            end_idx = i + config['seq_length']
            train_sequences[i] = x_train[i:end_idx]
            train_targets[i] = x_train[end_idx: end_idx + config['horizon_size'], 0]

        for i in range(n_val_sequences-config['horizon_size']):
            end_idx = i + config['seq_length']
            val_sequences[i] = x_val[i:end_idx]
            val_targets[i] = x_val[end_idx: end_idx + config['horizon_size'], 0]

        # TODO: Shuffle targets and sequences using the same permutation

        return train_sequences, train_targets, val_sequences, val_targets

    def invert_DAM_predictions(self, data):
        # transform new data according to the fitted data
        pass
