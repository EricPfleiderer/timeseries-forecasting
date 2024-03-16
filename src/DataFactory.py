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
        if config['data_space']['log_transform']:
            out_data = torch.log(out_data)

        # Split train / val early to avoid data leakages from differencing, normalization  and other transforms
        x_train = out_data[:int(config['settings']['train_val_split']*out_data.size(0))]
        x_val = out_data[int(config['settings']['train_val_split']*out_data.size(0)):]

        # Difference transforms
        # TODO: Add n-differencing to remove trends of nth order

        # Feature wise normalization
        for i in range(out_data.size(1)):
            normalizer = config['data_space'][f'normalizer_{i}']()
            np_norm_feat_train = normalizer.fit_transform(torch.unsqueeze(x_train[:, i], dim=1))
            np_norm_feat_val = normalizer.transform(torch.unsqueeze(x_val[:, i], dim=1))
            x_train[:, i] = torch.squeeze(torch.Tensor(np_norm_feat_train), dim=1)
            x_val[:, i] = torch.squeeze(torch.Tensor(np_norm_feat_val), dim=1)
            self.normalizers.append(normalizer)

        # Build training sequences and targets
        seq_len = config['model_space']['seq_length']
        horizon_size = config['model_space']['horizon_size']
        n_train_sequences = x_train.size(0) - seq_len
        n_val_sequences = x_val.size(0) - seq_len
        train_sequences = torch.empty(size=(n_train_sequences, seq_len, x_train.size(1)))
        train_targets = torch.empty(size=(n_train_sequences, horizon_size))
        val_sequences = torch.empty(size=(n_val_sequences, seq_len, x_val.size(1)))
        val_targets = torch.empty(size=(n_val_sequences, horizon_size))

        for i in range(n_train_sequences-horizon_size):
            end_idx = i + seq_len
            train_sequences[i] = x_train[i:end_idx]
            train_targets[i] = x_train[end_idx: end_idx + horizon_size, 0]

        for i in range(n_val_sequences-horizon_size):
            end_idx = i + seq_len
            val_sequences[i] = x_val[i:end_idx]
            val_targets[i] = x_val[end_idx: end_idx + horizon_size, 0]

        # TODO: Shuffle targets and sequences using the same permutation
        return train_sequences, train_targets, val_sequences, val_targets

    def recover_scale(self, data):
        # TODO: Build inverse pipeline to recover original data scale
        pass
