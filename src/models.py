import torch
from ray import tune


class BasicLSTM(torch.nn.Module):
    def __init__(self, n_features, seq_length, n_hidden, n_layers, horizon_size):
        super(BasicLSTM, self).__init__()
        self.seq_len = seq_length  # Length of input sequences
        self.n_features = n_features  # Number of features in input sequences
        self.n_hidden = n_hidden  # Number of features in hidden states
        self.n_layers = n_layers  # Number of LSTM layers (stacked)
        self.horizon_size = horizon_size  # Number of steps predicted by the model

        self.l_lstm = torch.nn.LSTM(input_size=n_features,
                                    hidden_size=self.n_hidden,
                                    num_layers=self.n_layers,
                                    batch_first=True)
        self.l_linear = torch.nn.Linear(self.n_hidden * self.seq_len, self.horizon_size)

    def init_hidden(self, batch_size, device):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden).to(device)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden).to(device)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        lstm_out, self.hidden = self.l_lstm(x, self.hidden)
        x = lstm_out.contiguous().view(batch_size, -1)
        return self.l_linear(x)

    @staticmethod
    def get_hyperspace():
        return {
            'seq_length': tune.randint(24, 14*24),
            'n_hidden': tune.randint(20, 100),
            'n_layers': tune.randint(2, 10),
            'horizon_size': tune.randint(24, 14*24),
        }
