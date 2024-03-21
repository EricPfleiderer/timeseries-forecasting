import logging

import pandas as pd
import torch
from copy import copy
import numpy as np


class DataFactory:

    def __init__(self):
        self.pd_data = pd.read_excel("data/energy.xlsx")
        self.np_data = self.pd_data.iloc[:, 2:].to_numpy()
        self.np_data = self.interpolate_nans(self.np_data)
        self.data = torch.Tensor(self.np_data)
        self.normalizers = []
        self.config = None

    @staticmethod
    def interpolate_nans(np_data):

        # Data should have shape [n_samples, n_features]

        def nan_finder(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        for i in range(np_data.shape[1]):
            nans, x = nan_finder(np_data[:, i])
            np_data[nans, i] = np.interp(x(nans), x(~nans), np_data[~nans, i])

        DataFactory.assert_clean_data(torch.tensor(np_data))

        return np_data

    @ staticmethod
    def assert_clean_data(data):
        assert not torch.any(torch.isnan(data))
        assert not torch.any(torch.isinf(data))

    def generate_datasets(self, config):

        header_padding = ''.join([' ' for _ in range(50)])

        logging.info(f'Generating datasets...')

        self.config = config
        out_data = copy(self.data)
        logging.info(f'Raw data:             ' + header_padding + DataFactory.describe_tensor(out_data))

        logging.info('Performing power transforms')
        if config['data_space']['log_transform']:
            out_data = torch.log(out_data)

        DataFactory.assert_clean_data(out_data)

        logging.info('Splitting data into training and validation sets')
        # Split train / val early to avoid data leakages from differencing, normalization  and other transforms
        x_train = out_data[:int(config['settings']['train_val_split']*out_data.size(0))]
        x_val = out_data[int(config['settings']['train_val_split']*out_data.size(0)):]

        logging.info('Performing difference transforms')
        # TODO: Add n-differencing to remove trends of nth order

        logging.info('Performing normalization transforms')
        for i in range(out_data.size(1)):
            normalizer = config['data_space'][f'normalizer_{i}']()
            np_norm_feat_train = normalizer.fit_transform(torch.unsqueeze(x_train[:, i], dim=1))
            np_norm_feat_val = normalizer.transform(torch.unsqueeze(x_val[:, i], dim=1))
            x_train[:, i] = torch.squeeze(torch.Tensor(np_norm_feat_train), dim=1)
            x_val[:, i] = torch.squeeze(torch.Tensor(np_norm_feat_val), dim=1)
            self.normalizers.append(normalizer)

        logging.info(f'Training data:        ' + header_padding + DataFactory.describe_tensor(x_train))
        logging.info(f'Validation data:      ' + header_padding + DataFactory.describe_tensor(x_val))

        DataFactory.assert_clean_data(out_data)

        logging.info('Building final sequences...')
        seq_len = config['model_space']['seq_length']
        horizon_size = config['model_space']['horizon_size']
        n_train_sequences = x_train.size(0) - seq_len - horizon_size
        n_val_sequences = x_val.size(0) - seq_len - horizon_size
        train_sequences = torch.empty(size=(n_train_sequences, seq_len, x_train.size(1)))
        train_targets = torch.empty(size=(n_train_sequences, horizon_size))
        val_sequences = torch.empty(size=(n_val_sequences, seq_len, x_val.size(1)))
        val_targets = torch.empty(size=(n_val_sequences, horizon_size))

        for i in range(n_train_sequences):
            end_idx = i + seq_len
            train_sequences[i] = x_train[i:end_idx]
            train_targets[i] = x_train[end_idx: end_idx + horizon_size, 0]

        for i in range(n_val_sequences):
            end_idx = i + seq_len
            val_sequences[i] = x_val[i:end_idx]
            val_targets[i] = x_val[end_idx: end_idx + horizon_size, 0]

        logging.info(f'Validation data:      ' + header_padding + DataFactory.describe_tensor(x_val))
        logging.info(f'Training sequences:   ' + header_padding + DataFactory.describe_tensor(train_sequences))
        logging.info(f'Training targets:     ' + header_padding + DataFactory.describe_tensor(train_targets))
        logging.info(f'Validation sequences: ' + header_padding + DataFactory.describe_tensor(val_sequences))
        logging.info(f'Validation targets:   ' + header_padding + DataFactory.describe_tensor(val_targets))
        # TODO: Shuffle targets and sequences using the same permutation
        return train_sequences, train_targets, val_sequences, val_targets

    @staticmethod
    def describe_tensor(tensor_data):

        if len(tensor_data.size()) != 2 and len(tensor_data.size()) != 3:
            raise ValueError('The describe_tensor method only supports 2d or 3d inputs')

        shape_chars = 25

        output = f'| Shape: {tensor_data.size(0)} x {tensor_data.size(1)}'

        if len(tensor_data.size()) == 3:
            output += f' x {tensor_data.size(2)}'

        output += ''.join([' ' for _ in range(shape_chars - len(output))]) + '|'
        output += ' min: {0:.6f} '.format(round(float(torch.min(tensor_data)), 6)) + '|'
        output += ' max: {0:.6f} '.format(round(float(torch.max(tensor_data)), 6)) + '|'

        return output

    def recover_scale(self, data):
        # TODO: Build inverse pipeline to recover original data scale
        pass
