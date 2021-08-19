import copy
import os

import ray
import torch


@ray.remote
class SharedStorage:
    """
    Class which runs in a dedicated thread to store the network weights and some information to disk.
    """

    def __init__(self, checkpoint, config):
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)

    # Saves checkpoint to disk in game's results directory.
    def save_checkpoint(self, path=None):
        if not path:
            path = os.path.join(self.config.results_path, "model.checkpoint")

        torch.save(self.current_checkpoint, path)

    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)

    # Gets checkpoint info/data. keys are used to determine what info to get, so like "total_reward" or "num_played_games" or "terminate" or something.
    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    # Sets checkpoint info. keys are used to define what info to get, so like "total_reward" or "num_played_games" or "terminate" or something.
    # FIXME: Does this save to disk or RAM? I think to disk?
    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError

### That was easy! Head back to trainer.py now!