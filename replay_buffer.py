import copy
import time

import numpy
import ray
import torch

import models


@ray.remote
class ReplayBuffer:
    """
    Class which runs in a dedicated thread to store played games and, from it, generate a batch.
    """

    def __init__(self, initial_checkpoint, initial_buffer, config):
        self.config = config
        self.buffer = copy.deepcopy(initial_buffer)
        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]  # So as far as I can tell, the difference between a step and a game is that a step is used by the trainer to train the model and contains multiple games depending on the checkpoint_interval.
        self.total_samples = sum([len(game_history.root_values) for game_history in self.buffer.values()])  # I'm not sure where .values() is defined.
        # If we're not starting training from scratch, e.g. we loaded a pretrained (or partially trained) model from disk, let the user know.
        if self.total_samples != 0:
            print(f"Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n")

        numpy.random.seed(self.config.seed)

    ### Just wanted to create the init for this thing, we'll come back to it later. In the meantime, head back to muzero.py!

    ### Hey, if you're reading this, you just came over from muzero.py and self_play.py should be fully implemented. Now let's finish replay_buffer.py so we can run this thing.

    # Game history is defined in self_play.py
    def save_game(self, game_history, shared_storage=None):
        # If we're doing Prioritized Replay, then either copy the existing priorities or set the initial priorities if there are none.
        # Prioritized replay is explained in trainer.py in the function continuous_update_weights()
        if self.config.PER:
            if game_history.priorities is not None:
                # "Avoid read only array when loading replay buffer from disk"
                # FIXME: What? This appears to be a shallow copy, so how is this not a read-only array?
                game_history.priorities = numpy.copy(game_history.priorities)
            else:
                # Initial priorities for the prioritized replay (See paper appendix Training)
                priorities = []
                for i, root_value in enumerate(game_history.root_values):
                    priority = (numpy.abs(root_value - self.compute_target_value(game_history, i)) ** self.config.PER_alpha)  # Use some voodoo to set the priority of the root value at index i.
                    priorities.append(priority)  # Add that priority to list of priorities.

                game_history.priorities = numpy.array(priorities, dtype="float32")  # Set game_history.priorities to a numpy array created from the list of priorities we just made.
                game_history.game_priority = numpy.max(game_history.priorities)  # Set the game_priority to the max value of the priorities. What game_priority used for?

        self.buffer[self.num_played_games] = game_history  # Assign game_history to the spot in the buffer correlating to the current game number, aka num_played_games.
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)  # So "steps" seem to be individual moves made in the game. Alright, cool, that's something I was confused about earlier.
        self.total_samples += len(game_history.root_values)

        # If the config-defined replay buffer size is smaller than the actual size of the buffer, delete the items in the buffer that go over replay_buffer_size.
        if self.config.replay_buffer_size < len(self.buffer):
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

        # If we have a shared storage, save num_played_games and num_played_steps to it. shared_storage=None by default.
        if shared_storage:
            shared_storage.set_info.remote("num_played_games", self.num_played_games)
            shared_storage.set_info.remote("num_played_steps", self.num_played_steps)

    def get_buffer(self):
        return self.buffer

    def get_batch(self):
        (  # Initialize all of the batches as lists.
            index_batch,
            observation_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_batch,
            gradient_scale_batch,
        ) = ([], [], [], [], [], [], [])
        weight_batch = [] if self.config.PER else None

        # So for each game we're sampling, add samples to each batch.
        for game_id, game_history, game_prob in self.sample_n_games(self.config.batch_size):
            game_pos, pos_prob = self.sample_position(game_history)  # Sample position from a game in the game history.

            values, rewards, policies, actions = self.make_target(game_history, game_pos)  # What are targets? Optimal values?

            # Append each batch with respective data.
            index_batch.append([game_id, game_pos])
            observation_batch.append(game_history.get_stacked_observations(game_pos, self.config.stacked_observations))  # Append the observation batch with stacked observations.
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            gradient_scale_batch.append([min(self.config.num_unroll_steps, len(game_history.action_history) - game_pos)] * len(actions))  # I'm not sure what this is used for.
            if self.config.PER:
                weight_batch.append(1 / (self.total_samples * game_prob * pos_prob))

        if self.config.PER:
            weight_batch = numpy.array(weight_batch, dtype="float32") / max(weight_batch)

        # I'm not sure what any of this means.
        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1

        # Return a tuple that contains the index batch and a tuple of all the other batches?
        # So return the index batch and the next batch, which is a batch of all the other batches. Got it.
        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                value_batch,
                reward_batch,
                policy_batch,
                weight_batch,
                gradient_scale_batch,
            ),
        )

    # Sample a game from the game history to be used in reanalyze for something.
    def sample_game(self, force_uniform=False):
        """
        Sample game from buffer either uniformly or according to some priority. Sample what from the game? Oh, sample A game, the whole thing, for use in reanalyze.
        See paper appendix Training.
        Return the game id, the game from the buffer, and the game_prob, which is either the game's priorities is we're using PR or None if we're not.
        """
        game_prob = None

        if self.config.PER and not force_uniform:
            game_probs = numpy.array([game_history.game_priority for game_history in self.buffer.values()], dtype="float32")
            game_probs /= numpy.sum(game_probs)
            game_index = numpy.random.choice(len(self.buffer), p=game_probs)
            game_prob = game_probs[game_index]
        else:
            game_index = numpy.random.choice(len(self.buffer))

        game_id = self.num_played_games - len(self.buffer) + game_index

        return game_id, self.buffer[game_id], game_prob

    # Sample n games from game_history so we can get a batch based on them.
    # Based on the coding style, it looks to me like someone else coded this part. Mainly because things aren't spread out needlessly over multiple lines.
    def sample_n_games(self, n_games, force_uniform=False):
        # Either sample games based on prioritized replay or randomly.
        if self.config.PER and not force_uniform:
            game_id_list = []
            game_probs = []  # A list of game priorities
            # Get list of game IDs and the respective priorities of those games.
            for game_id, game_history in self.buffer.items():
                game_id_list.append(game_id)
                game_probs.append(game_history.game_priority)
            # Do some voodoo with the game priorities, create a dictionary of them, then select games with them.
            game_probs = numpy.array(game_probs, dtype="float32")
            game_probs /= numpy.sum(game_probs)
            game_prob_dict = dict([(game_id, prob) for game_id, prob in zip(game_id_list, game_probs)])
            selected_games = numpy.random.choice(game_id_list, n_games, p=game_probs)  # Setting p=game_probs lets us select games based on their priorities.
        # If we're not using prioritized replay, just select the games randomly I think.
        else:
            selected_games = numpy.random.choice(list(self.buffer.keys()), n_games)
            game_prob_dict = {}
        # Create list of tuples, each containing a game ID, the respective game in the buffer, and the game's priority from game_prob_dict.
        ret = [(game_id, self.buffer[game_id], game_prob_dict.get(game_id)) for game_id in selected_games]
        return ret

    # Used in get_batch, sample a random position from any game in game_history.
    def sample_position(self, game_history, force_uniform=False):
        """
        Sample a random position from any game in game_history, either uniformly or according to some priority.
        See paper appendix Training.
        """
        position_prob = None
        # If doing prioritized replay, use priorities to pick a position from one of the games and return that position's priority, else return a random position.
        if self.config.PER and not force_uniform:
            position_probs = game_history.priorities / sum(game_history.priorities)  # ??? Voodoo on the priorities? for what purpose?
            position_index = numpy.random.choice(len(position_probs), p=position_probs)
            position_prob = position_probs[position_index]
        else:
            position_index = numpy.random.choice(len(game_history.root_values))

        return position_index, position_prob

    # Used in reanalyze, updates priorities using game_history and saves game_history to game's spot in buffer.
    def update_game_history(self, game_id, game_history):
        # The element could have been removed since its selection and update? What? Maybe with the context of how this is used, this will make more sense.
        if next(iter(self.buffer)) <= game_id:
            if self.config.PER:
                # Avoid read only array when loading replay buffer from disk? What? Just like earlier, won't this only create a shallow copy?
                game_history.priorities = numpy.copy(game_history.priorities)
            self.buffer[game_id] = game_history  # Set game in buffer to game history?

    def update_priorities(self, priorities, index_info):
        """
        Update game and position priorities with priorities calculated during the training.
        See Distributed Prioritized Experience Replay https://arxiv.org/abs/1803.00933
        """
        for i in range(len(index_info)):
            game_id, game_pos = index_info[i]

            # Like above, the element could have been removed since its selection and training
            if next(iter(self.buffer)) <= game_id:
                # Update the priorities of each of the individual positions in the game, so muzero can decide which positions are best to learn from I think.
                priority = priorities[i, :]  # [i, :] takes a slice from priorities I think: https://stackoverflow.com/questions/54469789/what-does-i-mean-in-python/54469843
                start_index = game_pos
                end_index = min(game_pos + len(priority), len(self.buffer[game_id].priorities))
                # Set the priorities of the game from the start index to the end index as the input priorities from the start index to the end index.
                self.buffer[game_id].priorities[start_index:end_index] = priority[: end_index - start_index]  # FIXME: I'm not sure if the exact syntax he used here matters. I hope it doesn't.

                # Update the priority of the game itself, so muzero can decide if it should train on it vs other games.
                self.buffer[game_id].game_priority = numpy.max(self.buffer[game_id].priorities)

    # Used in calculating priorities for prioritized replay
    # Target values I think are values that the value network SHOULD predict based on the game state? They're returned by get_batch() as the action_batch, and get_batch() is only used in trainer.py, so I'm pretty sure they're used for training.
    # Anyways, this has some voodoo that I don't entirely understand yet. I'm kind of behind schedule (it's thursday and I wanted to have this whole thing done by the end of the week), and I've got a headache right now, so I'll pick this apart in further detail later.
    # index is the index of the position/state that we're computing the target value of (at least, I'm pretty sure it's a position, since we use it to find a root value)
    def compute_target_value(self, game_history, index):
        # The value target is the discounted root value of the search tree td_steps into the future, plus the discounted sum of all rewards until then.
        bootstrap_index = index + self.config.td_steps  # td_steps is the number of steps in the future to take into account when calculating the target value.
        if bootstrap_index < len(game_history.root_values):
            root_values = (game_history.root_values if game_history.reanalysed_predicted_root_values is None
                           else game_history.reanalysed_predicted_root_values)
            last_step_value = (root_values[bootstrap_index] if game_history.to_play_history[bootstrap_index] == game_history.to_play_history[index]
                               else -root_values[bootstrap_index])
            value = last_step_value * self.config.discount ** self.config.td_steps
        else:
            value = 0

        # Use rewards from game history to calculate value.
        for i, reward in enumerate(game_history.reward_history[index + 1 : bootstrap_index + 1]):
            # The value is oriented from the perspective of the current player
            value += (reward if game_history.to_play_history[index] == game_history.to_play_history[index + i] else -reward) * self.config.discount ** i

        return value

    # State index is the index of the state/position we start generating targets from.
    def make_target(self, game_history, state_index):
        """
        Generate targets for every unroll(?) steps(?). For each unrolled step?
        So I'm pretty sure targets are used to train the neural networks, since make_target() is only used by get_batch() and get_batch() is only used by trainer.py.
        :return: target_values, target_rewards, target_policies, actions (actions we've already taken starting from a position we're sampling from)
        """
        # Initialize lists of values, rewards, policies, and actions to be put in batches to train on.
        target_values, target_rewards, target_policies, actions = [], [], [], []
        # From state_index to (state_index + num_unroll_steps + 1), add stuff to lists initialized above.
        # num_unroll_steps is the number of game moves to keep for every batch element.
        for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
            value = self.compute_target_value(game_history, current_index)  # Compute target value of current state.

            if current_index < len(game_history.root_values):
                # Append all lists with relevant data from game_history.
                target_values.append(value)
                target_rewards.append(game_history.reward_history[current_index])
                target_policies.append(game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index])
            elif current_index == len(game_history.root_values):
                # Since we don't yet know the value of the current game state, append target_values with 0 and target_policies with... some voodoo?
                target_values.append(0)
                target_rewards.append(game_history.reward_history[current_index])
                # Uniform policy? What does that mean?
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(game_history.action_history[current_index])  # FIXME: Oh jesus, I almost forgot to add this line. I'll have to compare my code with his after I finish copying it to make sure I didn't mess anything up.
            else:
                # States past the end of games are treated as absorbing states.
                # Wtf is an absorbing state? Whatever, since we don't know anything about these states, set target_values and target_rewards to 0 and do the same voodoo with target_policies.
                # As for the future actions, just assume we'll take random actions?
                target_values.append(0)
                target_rewards.append(0)
                # Uniform policy again.
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(numpy.random.choice(self.config.action_space))

        return target_values, target_rewards, target_policies, actions


@ray.remote
class Reanalyze:
    """
    Class which runs in a dedicated thread to update the replay buffer with fresh information. See paper appendix Reanalyze.
    """

    def __init__(self, initial_checkpoint, config):
        # This init is basically the same as SelfPlay's, except we get the number of reanalyze games.
        self.config = config

        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.reanalyse_on_gpu else "cpu"))
        self.model.eval()

        self.num_reanalysed_games = initial_checkpoint["num_reanalysed_games"]

        ### Ayy, if you've gotten past the part where I've implemented self_play.py, then ignore the comment below.
        ### And with this implemented, the code runs once again! Back to muzero.py!

    def reanalyse(self, replay_buffer, shared_storage):
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        # Where all the reanalyzing is done.
        while ray.get(shared_storage.get_info.remote("training_step")) < self.config.training_steps and not ray.get(shared_storage.get_info.remote("terminate")):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))  # Set weights to weights in shared_storage

            game_id, game_history, _ = ray.get(replay_buffer.sample_game.remote(force_uniform=True))  # The underscore is used to indicate that part of the function's result is being deliberately ignored, it's essentially a "throwaway" variable.

            # I don't know why this if statement is necessary, since use_last_model_value MUST be true in order for this code to be run by muzero.py.
            # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
            if self.config.use_last_model_value:
                # Get observations from game history.
                observations = [game_history.get_stacked_observations(i, self.config.stacked_observations)
                                for i in range(len(game_history.root_values))]
                # Convert observations from game history to torch tensor.
                observations = torch.tensor(observations).float().to(next(self.model.parameters()).device)  # FIXME: Hope the parentheses aren't necessary here...
                # Use representation function to convert observations to hidden state, then use that hidden state to predict state values.
                values = models.support_to_scalar(self.model.initial_inference(observations)[0], self.config.support_size)
                game_history.reanalysed_predicted_root_values = (torch.squeeze(values).detach().cpu().numpy())

            replay_buffer.update_game_history.remote(game_id, game_history)
            self.num_reanalysed_games += 1
            shared_storage.set_info.remote("num_reanalysed_games", self.num_reanalysed_games)

            ### Alright, great, we're done with the entire replay buffer! Now lets head over to trainer.py and finish that.