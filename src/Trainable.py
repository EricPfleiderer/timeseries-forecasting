from typing import Optional, Dict

import torch
import ray.tune as tune
from ray.tune import PlacementGroupFactory

from src.DataFactory import DataFactory
import os
import sys


class Trainable(tune.Trainable):

    def setup(self, config: Dict):
        self.config = config

        # Device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Data
        self.data_factory = DataFactory()
        self.x_train, self.y_train, self.x_val, self.y_val = self.data_factory.generate_datasets(config)

        # Model
        self.model = config['settings']['model'](config['settings']['n_features'], **config['model_space'])
        self.model.to(self.device)

        # Training
        self.criterion = torch.nn.MSELoss()
        self.optimizer = config['training_space']['optimizer'](self.model.parameters(), **config['training_space']['optimizer_space'])

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Dict]:
        # Subclasses should override this to implement save().

        """
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, "w") as f:
            f.write(json.dumps({"timestep": self.timestep}))
        """

        return {"checkpoint_dir": checkpoint_dir}

    def load_checkpoint(self, checkpoint: Optional[Dict]):
        # Subclasses should override this to implement restore().

        """
        path = os.path.join(checkpoint["checkpoint_dir"], "checkpoint")
        with open(path, "r") as f:
            self.timestep = json.loads(f.read())["timestep"]
        """
        pass

    def step(self):

        loss = None
        batch_size = self.config['training_space']['batch_size']

        for t in range(self.config['training_space']['n_epochs']):
            for b in range(0, len(self.x_train), batch_size):
                inpt = self.x_train[b:b + batch_size, :, :]
                target = self.y_train[b:b + batch_size, :]

                x_batch = inpt.clone().detach().to(self.device)
                y_batch = target.clone().detach().to(self.device)

                self.model.init_hidden(x_batch.size(0), self.device)
                output = self.model(x_batch)

                # TODO: Recover original data scale to ensure fair comparison between different models (log)
                loss = self.criterion(output.view(-1), y_batch.view(-1))

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()


        return {"episode_reward_mean": loss.item()}

    def reset_config(self, new_config: Dict):
        # Resets configuration without restarting the trial.
        pass

    def cleanup(self):
        # Subclasses should override this for any cleanup on stop.
        pass

    """
    @classmethod
    def default_resource_request(cls, config):
        return PlacementGroupFactory([{"CPU": 2}, {"CUDA": 1/4}])
    """



