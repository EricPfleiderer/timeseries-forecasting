import logging
from typing import Optional, Dict

import torch
import ray.tune as tune
from ray.air.integrations.wandb import setup_wandb
from ray.tune import PlacementGroupFactory

from src.DataFactory import DataFactory
from src.enums import Colors


class Trainable(tune.Trainable):

    def setup(self, config: Dict):
        logging.info(Colors.BOLD.value + f'SET UP START: Trial {self.trial_id}' + Colors.END.value)
        self.config = config
        self.device = None

        '''
        self.wandb = setup_wandb(
            config,
            trial_id=self.trial_id,
            trial_name=self.trial_name,
            group="Example",
            project='timeseries-forecasting',
        )
        '''

        self.data_factory = DataFactory()
        self.x_train, self.y_train, self.x_val, self.y_val = self.data_factory.generate_datasets(config)

        self.model = config['settings']['model'](self.x_train.size(-1), **config['model_space'])

        self.criterion = torch.nn.MSELoss()
        self.optimizer = config['training_space']['optimizer'](self.model.parameters(), **config['training_space']['optimizer_space'])

        logging.info(Colors.BOLD.value + f'SET UP END: Trial {self.trial_id}' + Colors.END.value)

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Dict]:
        # Subclasses should override this to implement save().
        return {"checkpoint_dir": checkpoint_dir}

    def load_checkpoint(self, checkpoint: Optional[Dict]):
        # Subclasses should override this to implement restore().
        pass

    def step(self):
        logging.info(Colors.BOLD.value + f'STEP #{self.iteration} START: Trial {self.trial_id}' + Colors.END.value)
        batch_size = self.config['training_space']['batch_size']

        # Device
        # TODO: Debug cuda (ray not making cuda available within Trainable)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model.to(self.device)

        logging.info(f'Sending model and data to {device}')

        # Logging
        n_train_batches = self.x_train.size(0) // batch_size
        n_val_batches = self.x_val.size(0) // batch_size
        train_logs_granularity = 10
        val_logs_granularity = 5
        train_logs_idx = [int(n_train_batches / train_logs_granularity * i) for i in range(1, 1 + train_logs_granularity)]
        val_logs_idx = [int(n_val_batches / val_logs_granularity * i) for i in range(1, 1 + val_logs_granularity)]

        # Training
        mean_train_loss = 0
        mean_val_loss = 0
        for b in range(0, len(self.x_train), batch_size):

            inpt = self.x_train[b:b + batch_size, :, :]
            target = self.y_train[b:b + batch_size, :]

            x_batch = inpt.clone().detach().to(self.device)
            y_batch = target.clone().detach().to(self.device)

            self.model.init_hidden(x_batch.size(0), self.device)
            output = self.model(x_batch)

            DataFactory.assert_clean_data(output)

            train_loss = self.criterion(output.view(-1), y_batch.view(-1))
            train_loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            mean_train_loss += train_loss.item()

            batch_i = b//batch_size
            if batch_i in train_logs_idx:
                logging.info(f'[Training]    {Trainable.get_batch_progress(batch_i, n_train_batches, train_loss)}')

        mean_train_loss /= n_train_batches
        logging.info('[Training]    ' + Colors.UNDERLINE.value + 'Mean loss: {0:.6f}'.format(mean_train_loss) + Colors.END.value)

        # Validation
        for b in range(0, len(self.x_val), batch_size):
            inpt = self.x_val[b:b + batch_size, :, :]
            target = self.y_val[b:b + batch_size, :]

            x_batch = inpt.clone().detach().to(self.device)
            y_batch = target.clone().detach().to(self.device)

            self.model.init_hidden(x_batch.size(0), self.device)
            output = self.model(x_batch)

            # TODO: Compute val loss on original data scale to ensure fairness across trials
            val_loss = self.criterion(output.view(-1), y_batch.view(-1))

            mean_val_loss += val_loss.item()

            batch_i = b // batch_size
            if batch_i in val_logs_idx:
                logging.info(f'[Validation]  {Trainable.get_batch_progress(batch_i, n_val_batches, val_loss)}')

        mean_val_loss /= n_val_batches
        logging.info('[Validation]  ' + Colors.UNDERLINE.value + 'Mean loss: {0:.6f}'.format(mean_val_loss) + Colors.END.value)

        logging.info(Colors.BOLD.value + f'STEP #{self.iteration} END: Trial {self.trial_id}' + Colors.END.value)
        metrics = {'mean_train_loss': mean_train_loss, 'mean_val_loss': mean_val_loss}
        # self.wandb.log(metrics)
        return metrics

    @staticmethod
    def get_batch_progress(batch_i, n_batches, loss):
        progress = int((batch_i / n_batches) * 100)
        loading_bar = ''.join(['=' for _ in range(progress)]) + ">" + ''.join(['-' for _ in range(100 - progress)])
        num_padding = ''.join(['0' for _ in range(3 - len(str(batch_i)))])
        denum_padding = ''.join(['0' for _ in range(3 - len(str(n_batches)))])

        return f'{num_padding}{batch_i}/{denum_padding}{n_batches} |{loading_bar}|' + ' loss: {0:.6f}'.format(round(loss.item(), 6))

    def cleanup(self):
        # Subclasses should override this for any cleanup on stop.
        pass

    @classmethod
    def default_resource_request(cls, config):
        return PlacementGroupFactory([{"CPU": 2}, {"GPU": 1/4}])



