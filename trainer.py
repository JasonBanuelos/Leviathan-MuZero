import copy
import time

import ray
import numpy
import torch

import models


@ray.remote
class Trainer:
    """
    Class which runs in a dedicated thread to train a neural network and save it in the shared storage. Shared storage is defined in shared_storage.py, but I'm not going to implement that just yet.
    Use this class to define workers to train MuZero.

    Args:
        initial_checkpoint: The checkpoint we're training, could be from scratch or could be a checkpoint we loaded I think.
        config: The config of the game file.
        I'm not exactly sure what the types of those two args are.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Like in MuZero's init, set torch & numpy's RNG seed to whatever you specified in the game file.
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Set up the networks we're going to train.
        self.model = models.MuZeroNetwork(self.config)  # Set up the model's neural nets according the game config.
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))  # Set the model's weights as the weights from the checkpoint.
        self.model.to(torch.device("cuda" if self.config.train_on_gpu else "cpu"))  # If train_on_gpu is true in the config, then train in GPU, else train on CPU.
        self.model.train()  # This is torch's .train(), not ours. From torch.nn.Module, sets the module in training mode. Doesn't actually start training, though, which is kind of confusing.

        self.training_step = initial_checkpoint["training_step"]  # Sets our initial training step to whatever the checkpoint's is.

        # Let the user know the model isn't training on GPU if it's not.
        if "cuda" not in str(next(self.model.parameters()).device):  # .parameters() is from torch.nn.Module. .device is a string indicating, you guessed it, the device we're training on.
            print("You are not training on GPU.\n")

        # Initialize the optimizer, either SGD or Adam. Could potentially implement a different optimizer here, would be as easy as adding another elif with torch.optim.[your optimizer]
        if self.config.optimizer == "SGD":
            # This is another one of those things that he spread out over multiple lines that I think is easier to digest as a single line.
            # Spreading an initialization like this out over multiple lines is like spreading the words of a sentence out over multiple lines. It just makes it look more convoluted than it really is.
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.lr_init, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr_init, weight_decay=self.config.weight_decay)
        else:
            raise NotImplementedError(f"{self.config.optimizer} is not implemented. You can implement a new optimizer in trainer.py, or change the optimizer in the config file.")

        # Load the checkpoint's optimizer state, if there is one.
        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(copy.deepcopy(initial_checkpoint["optimizer_state"]))

    ### The next function uses the SharedStorage class defined in shared_storage.py. Since that file is short, I decided to go ahead and implement it now, so head on over there!
    ### It also uses the replay_buffer, but that file is huge so I'll wait until I get an error to implement it.

    def continuous_update_weights(self, replay_buffer, shared_storage):
        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:  # While num_played_games < 1, sleep.
            time.sleep(0.1)

        next_batch = replay_buffer.get_batch.remote()  # A batch of training data, this article explains it well: https://medium.com/applied-data-science/how-to-build-your-own-deepmind-muzero-in-python-part-3-3-ccea6b03538b

        # Training loop, runs for the number of training steps declared in the game config.
        while self.training_step < self.config.training_steps and not ray.get(shared_storage.get_info.remote("terminate")):
            index_batch, batch = ray.get(next_batch)  # index batch is a batch of indexes of games and game positions I think? Used in update_priorities.remote()
            next_batch = replay_buffer.get_batch.remote()
            self.update_lr()  # update_lr and update_weights are functions that we've defined below in this file.
            # All of the meat of the training loop is essentially in this one line, as update_weights() performs a single training step.
            priorities, total_loss, value_loss, reward_loss, policy_loss = self.update_weights(batch)  # The way he wrote this in his code is so confusing, I thought all this was part of update_lr().

            if self.config.PER:  # PER in the config file is a bool for whether or not we're doing Prioritized Replay, which is explained here: https://arxiv.org/abs/1803.00933
                # Save new priorities in the replay buffer. What are priorities? Here's what the paper has to say:
                # "...instead of sampling uniformly, we prioritize, to sample the most useful data more often. ...Priorities can be defined in various ways, depending on the learning algorithm."
                # Since we can often learn more from some experiences than from others, we can assign priorities to each experience tuple in the buffer, allowing us to select these experience to train more often.
                # So basically a priority seems to be a number that determines how much to prioritize an experience to learn from. Pretty cool!
                replay_buffer.update_priorities.remote(priorities, index_batch)

            # Save training progress to the shared storage
            # I'm not sure why, but in the config file we specify the number of training steps before "using the model for self-playing", so here we determine whether or not to save the model? Why not just save it every time?
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(models.dict_to_cpu(self.optimizer.state_dict()))
                    }
                )
                if self.config.save_model:
                    shared_storage.save_checkpoint.remote()
            # Save everything about the training process other than the model.
            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.param_groups[0]["lr"],  # With optimizer.param_groups we can specify that we want to retrieve the learning rate I think.
                    "total_loss": total_loss,  # We got these four losses above at the beginning of the loop.
                    "value_loss": value_loss,
                    "reward_loss": reward_loss,
                    "policy_loss": policy_loss,
                }
            )

            # Adjust the self play / training ratio to avoid over/underfitting. So I guess adjusting the ratio can make it self play or train more.
            if self.config.training_delay:
                time.sleep(self.config.training_delay)
            if self.config.ratio:
                while (  # A really large condition for a while loop that just puts the program to sleep for a bit.
                    self.training_step / max(1, ray.get(shared_storage.get_info.remote("num_played_steps"))) > self.config.ratio
                    and self.training_step < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)
            # FIXME: Wait, so where do we increment training_step?

    # Perform one training step. Returns priorities, total_loss, value_loss, reward_loss, policy_loss.
    def update_weights(self, batch):
        """
        Perform one training step.
        """

        # Get a LOT of things from the batch. Here is an image of those things: https://miro.medium.com/max/2000/1*49FI1Uw0p7B_64xvEThveA.png
        observation_batch, action_batch, target_value, target_reward, target_policy, weight_batch, gradient_scale_batch, = batch

        # Keep values as scalars for calculating the priorities for the prioritized replay
        target_value_scalar = numpy.array(target_value, dtype="float32")  # Convert target value from a tuple to a scalar. But wait, an array isn't a scalar, it's a vector, right?
        priorities = numpy.zeros_like(target_value_scalar)  # numpy.zeros_like creates an array of zeros with the same shape and type as a given array.

        device = next(self.model.parameters()).device  # We training on GPU or CPU?

        # If we're doing Prioritized Replay, convert the weight batch to a tensor and move it to the training device
        if self.config.PER:
            weight_batch = torch.tensor(weight_batch.copy()).float().to(device)

        # Convert all of the other batches to a tensor and move it to the training device.
        observation_batch = torch.tensor(observation_batch).float().to(device)
        action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)  # Not sure why we're unsqueezing this one.
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)
        gradient_scale_batch = torch.tensor(gradient_scale_batch).float().to(device)
        # Not sure what these are for, but they were commented out in the original code. I guess it's a description of the dimensions of these batches?
        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1, 1 (unsqueeze)
        # target_value: batch, num_unroll_steps+1
        # target_reward: batch, num_unroll_steps+1
        # target_policy: batch, num_unroll_steps+1, len(action_space)
        # gradient_scale_batch: batch, num_unroll_steps+1

        target_value = models.scalar_to_support(target_value, self.config.support_size)
        target_reward = models.scalar_to_support(target_reward, self.config.support_size)
        # target_value: batch, num_unroll_steps+1, 2*support_size+1
        # target_reward: batch, num_unroll_steps+1, 2*support_size+1

        ### Hey, if self_play.py and replay_buffer.py have been fully implemented ignore the comments below and keep reading this file. Let's finish it up!

        ### At this point, I realized that I hadn't actually called continuous_update_weights() in main yet, so although I had was running the code while writing this,
        ### nothing I had written in trainer.py was actually being tested except for the init, so I decided to go back to muzero.py and keep working on that so that the
        ### code I had written here would actually get tested so I could make sure that it worked. Go ahead and head back to where we left off in muzero.py!

        # Generate initial predictions of value, reward, policy, and next hidden_state
        value, reward, policy_logits, hidden_state = self.model.initial_inference(observation_batch)
        predictions = [(value, reward, policy_logits)]  # Initialize list of predictions
        # Generate subsequent predictions and append them to list of predictions.
        for i in range(1, action_batch.shape[1]):
            value, reward, policy_logits, hidden_state = self.model.recurrent_inference(hidden_state, action_batch[:, i])
            # Scale the gradient at the start of the dynamics function (See paper appendix Training)
            hidden_state.register_hook(lambda grad: grad * 0.5)
            predictions.append((value, reward, policy_logits))
        # predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 | 2*support_size+1 | 9 (according to the 2nd dim)

        # Compute initial losses
        value_loss, reward_loss, policy_loss = (0, 0, 0)
        value, reward, policy_logits = predictions[0]
        # Ignore reward loss for the first batch step, hence the underscore. Why?

        ### I went ahead and went down to the bottom of the file and defined update_lr() and loss_function(), since they're short and necessary for the meat of the trainer.

        current_value_loss, _, current_policy_loss = self.loss_function(
            value.squeeze(-1),
            reward.squeeze(-1),
            policy_logits,
            target_value[:, 0],
            target_reward[:, 0],
            target_policy[:, 0],
        )
        value_loss += current_value_loss
        policy_loss += current_policy_loss
        # Compute priorities for the prioritized replay
        pred_value_scalar = models.support_to_scalar(value, self.config.support_size).detach().cpu().numpy().squeeze()
        priorities[:, 0] = (numpy.abs(pred_value_scalar - target_value_scalar[:, 0]) ** self.config.PER_alpha)  # You know what, I'm not sure if those outer parentheses are entirely necessary, but better safe than sorry.

        # Compute subsequent losses
        for i in range(1, len(predictions)):
            value, reward, policy_logits = predictions[i]
            current_value_loss, current_reward_loss, current_policy_loss = self.loss_function(
                value.squeeze(-1),
                reward.squeeze(-1),
                policy_logits,
                target_value[:, i],
                target_reward[:, i],
                target_policy[:, i],
            )

            # Scale gradient by the number of unroll steps
            current_value_loss.register_hook(lambda grad: grad / gradient_scale_batch[:, i])
            current_reward_loss.register_hook(lambda grad: grad / gradient_scale_batch[:, i])
            current_policy_loss.register_hook(lambda grad: grad / gradient_scale_batch[:, i])

            value_loss += current_value_loss
            reward_loss += current_reward_loss
            policy_loss += current_policy_loss

            # Compute priorities for the prioritized replay, again
            pred_value_scalar = models.support_to_scalar(value, self.config.support_size).detach().cpu().numpy().squeeze()
            priorities[:, i] = (numpy.abs(pred_value_scalar - target_value_scalar[:, i]) ** self.config.PER_alpha)

        # Scale the value loss to avoid overfitting of the value function, paper recommends by 0.25 (See paper appendix Reanalyze)
        loss = value_loss * self.config.value_loss_weight + reward_loss + policy_loss
        if self.config.PER:
            # Correct PER bias by using importance-sampling (IS) weights
            loss *= weight_batch
        # Mean over batch dimension (pseudocode do a sum?)
        loss = loss.mean()

        # Optimize. The most important part, but, interestingly, the shortest.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return (
            priorities,
            # For log purpose
            loss.item(),
            value_loss.mean().item(),
            reward_loss.mean().item(),
            policy_loss.mean().item(),
        )

    ### Alright, we're 100% done with both the trainer and the replay buffer! Now let's go back to muzero.py and see if we can run this thing!

    def update_lr(self):
        """
        Update learning rate. Still don't quite understand the learning rate.
        :return: Void
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (self.training_step / self.config.lr_decay_steps)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    # The loss function, surprise surprise, calculates the losses. Losses are the things we're trying to minimize via training.
    @staticmethod
    def loss_function(value, reward, policy_logits, target_value, target_reward, target_policy):
        # Cross-entropy seems to have a better convergence than MSE. I don't know what that means.
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1)
        reward_loss = (-target_reward * torch.nn.LogSoftmax(dim=1)(reward)).sum(1)
        policy_loss = (-target_policy * torch.nn.LogSoftmax(dim=1)(policy_logits)).sum(1)
        # Bruh, why is the parentheses in the .sum() in the policy loss spread across multiple lines in his code when the last two aren't? Did he create this thing with some kind of code generator? If so, where can I get my hands on it?

        return value_loss, reward_loss, policy_loss