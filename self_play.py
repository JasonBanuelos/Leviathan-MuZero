import time
import math

import numpy
import ray
import torch

import models


@ray.remote
class SelfPlay:
    """
    Class which runs in a dedicated thread to play games and save them to the replay buffer.
    """

    def __init__(self, initial_checkpoint, Game, config, seed):
        self.config = config
        self.game = Game(seed)

        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize MuZero's neural nets
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()

        ### And that's it, pretty simple init.
        ### I tried running it again, but this time I got this error:
        ###     self.replay_buffer = replay_buffer.Reanalyse.options(
        ### AttributeError: module 'replay_buffer' has no attribute 'Reanalyse'
        ### So I went back to replay_buffer.py to implement the init for Reanalyze. By the way, he spelled reanalyze wrong lol.

    # This function is where the self play workers continuously play the game, saving the played games to the replay buffer for training.
    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        while (
            ray.get(shared_storage.get_info.remote("training_step")) < self.config.training_steps
            and not ray.get(shared_storage.get_info.remote("terminate"))
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            # If not testing, play the game and save it to the replay buffer, otherwise take the best action (no exploration) in test mode
            if not test_mode:
                game_history = self.play_game(  # Play the game
                    # Although I think this multi-line argument is ugly, I also think it would be even uglier on one line, unfortunately.
                    # This softmax voodoo magic basically alters the visit count distribution to ensure that the action selection becomes greedier as training progresses.
                    # The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.
                    self.config.visit_softmax_temperature_fn(
                        trained_steps=ray.get(
                            shared_storage.get_info.remote("training_step")
                        )
                    ),
                    self.config.temperature_threshold,
                    False,
                    "self",
                    0
                )
                replay_buffer.save_game.remote(game_history, shared_storage)  # Save the game to the replay buffer
            else:
                # Take the best action (no exploration) in test mode
                game_history = self.play_game(
                    0,
                    self.config.temperature_threshold,
                    False,
                    "self" if len(self.config.players) == 1 else self.config.opponent,
                    self.config.muzero_player  # This is a number indicating the turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second).
                )
                # Save game/episode to shared storage
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "total_reward": sum(game_history.reward_history),
                        "mean_value": numpy.mean([value for value in game_history.root_values if value])
                    }
                )
                # If there's more than 1 players, save both MuZero's reward and the opponents reward, presumably to compare the two.
                if 1 < len(self.config.players):
                    shared_storage.set_info.remote(
                        {
                            "muzero_reward": sum(
                                # Sum rewards from reward history where the player is muzero.
                                reward for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1] == self.config.muzero_player
                            ),
                            "opponent_reward": sum(
                                # Sum rewards from reward history where the player is NOT muzero.
                                # But wait, what if the game is a 3-player game? Will ALL of muzero's opponents' rewards be summed together? That seems unfair to muzero.
                                reward for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1] != self.config.muzero_player
                            )
                        }
                    )

            # Similar to what we saw in trainer.py, here we adjust the self play / training ratio to avoid over/underfitting.
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)
            if not test_mode and self.config.ratio:
                while (  # Another really large condition for a while loop that just puts the program to sleep for a bit.
                    ray.get(shared_storage.get_info.remote("training_step")) / max(1, ray.get(shared_storage.get_info.remote("num_played_steps"))) < self.config.ratio
                    and ray.get(shared_storage.get_info.remote("training_step")) < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

        self.close_game()

    def play_game(self, temperature, temperature_threshold, render, opponent, muzero_player):
        """
        Play one game with actions based on Monte Carlo Tree Search at each move.

        :param temperature:             Used in selecting the action. From the paper, "visit count distribution is parametrized using a temperature" (Appendix D Data Generation)
        :param temperature_threshold:   Also used in selecting the action, I think its the max value that the temperature can be?
        :param render:                  Boolean, whether or not to render the game
        :param opponent:                The opponent to play against, either "self" or self.config.opponent
        :param muzero_player:           The trained muzero model to play against. FIXME: If muzero_player is 0 and opponent is not self, what happens?
        :return:                        game_history
        """

        game_history = GameHistory()
        observation = self.game.reset()
        game_history.action_history.append(0)  # Append action_history with 0? Why?
        ### Since this thing depends on the GameHistory class, I went down and implemented that. It's not too long.
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())  # to_play() determines the current player. It seems to be defined in both the game file and the Node class, but sometimes not in the game file? cartpole doesn't define to_play, so what happens there? is this just for multiplayer games?

        done = False

        # Render initial state of game/environment.
        if render:
            self.game.render()

        with torch.no_grad():  # with statement in Python is used in exception handling. Not sure what torch.no_grad is used for in this though.
            # Main gameplay loop.
            while not done and len(game_history.action_history) <= self.config.max_moves:
                # Make sure observation is 3 dimensional? Does it have to be 3 dimensional? What if the observation is a string, a one-dimensional array of characters?
                # Ohhh, I think the observation has to be 3 dimensional because the resnet is a 3d convolutional network? Can we change it to have a one dimensional input, though?
                assert (len(numpy.array(observation).shape) == 3), f"Observation should be 3 dimensional instead of {len(numpy.array(observation).shape)} dimensional. Got observation of shape: {numpy.array(observation).shape}"
                # Make sure the observation shape matches the observation_shape defined in MuZeroConfig. Makes sense.
                assert (numpy.array(observation).shape == self.config.observation_shape), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(observation).shape}."
                stacked_observations = game_history.get_stacked_observations(-1, self.config.stacked_observations)  # config.stacked_observations is the number of previous observations and previous actions to add to the current observation

                # Choose the action
                # If it's playing against itself, use select_action(), if it's playing against a preprogrammed or human opponent, use select_opponent_action().
                if opponent == "self" or muzero_player == self.game.to_play():
                    root, mcts_info = MCTS(self.config).run(self.model, stacked_observations, self.game.legal_actions(), self.game.to_play(), True)  # Create and run the MCTS.
                    # We input the root node because select_action() gets the visit counts of all the root node's children.
                    action = self.select_action(root, temperature if not temperature_threshold or len(game_history.action_history) < temperature_threshold else 0)
                    # If we're rendering the game, also print out information about the MCTS.
                    if render:
                        print(f'Tree depth: {mcts_info["max_tree_depth"]}')
                        print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
                else:
                    action, root = self.select_opponent_action(opponent, stacked_observations)

                # After choosing the action, execute it in the environment and from it get the observation, reward, and whether or not we're done.
                observation, reward, done = self.game.step(action)

                # Print the action taken and render game state after action has been taken.
                if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()

                # Store tree search statistics of game to game_history.
                game_history.store_search_statistics(root, self.config.action_space)

                # Store next batch
                game_history.action_history.append(action)
                game_history.observation_history.append(observation)
                game_history.reward_history.append(reward)
                game_history.to_play_history.append(self.game.to_play())

        # Return game_history, which is saved in the replay buffer and shared storage. FIXME: Shared storage saves to disk, but what about the replay buffer? Is that our bottleneck?
        return game_history

    # Is this function really necessary? Why not just use self.game.close() in continuous_self_play()? It's only one line.
    def close_game(self):
        self.game.close()

    # Select the action for the opponent playing against muzero.
    def select_opponent_action(self, opponent, stacked_observations):
        """
        Select the action for the opponent playing against muzero.

        :param opponent:                The opponent playing against muzero.
        :param stacked_observations:
        :return:                        action
        """
        # Select action depending on the type of opponent.
        if opponent == "human":
            # Create a MuZero MCTS so that MuZero can suggest a move to the human player.
            root, mcts_info = MCTS(self.config).run(
                self.model,
                stacked_observations,
                self.game.legal_actions(),
                self.game.to_play(),
                True,
            )
            # Print out info about the MCTS, let MuZero suggest its move to the human player.
            print(f'Tree depth: {mcts_info["max_tree_depth"]}')
            print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
            print(f"Player {self.game.to_play()} turn. MuZero suggests {self.game.action_to_string(self.select_action(root, 0))}")
            return self.game.human_to_action(), root
        elif opponent == "expert":
            return self.game.expert_agent(), None
        elif opponent == "random":
            # Make sure legal_actions are valid.
            assert (self.game.legal_actions()), f"Legal actions should not be an empty array. Got {self.game.legal_actions()}."
            assert set(self.game.legal_actions()).issubset(set(self.config.action_space)), "Legal actions should be a subset of the action space."
            return numpy.random.choice(self.game.legal_actions()), None
        else:
            raise NotImplementedError(f'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random". "{opponent}" is not an implemented opponent.')

        # Select MuZero's action based on the visit counts of each node in the MCTS. But where do the neural nets come into play?
    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function in the config.

        :param node:            We will get all the visit counts for all of this node's child nodes so we can calculate the visit_count_distribution to select the action.
        :param temperature:     Some value used in the voodoo we use to determine the visit_count_distribution
        :return:                action
        """
        # Get visit counts for all of the input node's child nodes.
        visit_counts = numpy.array([child.visit_count for child in node.children.values()], dtype="int32")
        # Make a list of actions which have resulted in a node. Is this a list of all possible actions?
        actions = [action for action in node.children.keys()]  # node.children is a dictionary, where the child nodes are the values and each child node's key is the action taken to get that child node.
        # If temperature == 0, select the action of the node with the most visits, if temperature == infinity, select the action randomly. If the temperature is somewhere in between, use some voodoo with the temperature and visit counts to select the action.
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            # Some voodoo to get the distribution of the visit count of each action. See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = numpy.random.choice(actions, p=visit_count_distribution)  # The probability for each action is determined by the visit_count_distribution.

        return action


# This MCTS is game independent, so you can use it to play any game.
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of the search tree and traversing the tree according to the UCB formula until we reach a
    leaf node. (UCB = Upper-Confidence-Bound, which is some voodoo used in MCTS, see https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Monte_Carlo_Method)
    """

    # Not much in the config, the real meat is in run()
    def __init__(self, config):
        self.config = config

    # Runs the MCTS, usually called at the same time the MCTS is initialized. to_play is the current player. override_root_with is the node to override the root node with.
    # FIXME: I still don't entirely understand this thing and exactly how it's used. You'll see my comments expounding my confusion below.
    def run(self, model, observation, legal_actions, to_play, add_exploration_noise, override_root_with=None):
        """
        At the root of the search tree we use the representation function to obtain a hidden state of the current observation.
        We then run a Monte Carlo Tree Search using only the action sequences and the model learned by the network. Cool, right?
        """
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            # Set up and expand the root node.
            root = Node(0)  # Initialize the root node
            observation = (
                torch.tensor(observation)
                .float()
                .unsqueeze(0)
                .to(next(model.parameters()).device)
            )
            root_predicted_value, reward, policy_logits, hidden_state = model.initial_inference(observation)  # model.initial_inference uses the representation function to encode MuZero's first (aka initial) observation.
            ### models.support_to_scalar is used in here, but I hadn't implemented that yet, so let's go over to models.py and implement it now!
            root_predicted_value = models.support_to_scalar(root_predicted_value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            assert legal_actions, f"Legal actions should not be an empty array. Got {legal_actions}."  # Assert legal_actions exists.
            assert set(legal_actions).issubset(set(self.config.action_space)), "Legal actions should be a subset of the action space."
            root.expand(legal_actions, to_play, reward, policy_logits, hidden_state)

        if add_exploration_noise:
            root.add_exploration_noise(dirichlet_alpha=self.config.root_dirichlet_alpha, exploration_fraction=self.config.root_exploration_fraction)

        min_max_stats = MinMaxStats()  # So this is what holds the "min-max values of the tree", so I guess that means it holds the best actions the tree predicts you could take?

        max_tree_depth = 0  # Initialize the depth of the tree.
        # With the root set up, FIXME: run as many tree searches as we set for num_simulations? Really, we're running more than one MCTS? I'm not sure if that's awesome or unnecessary.
        # FIXME: Wait, I'm confused, are we running MULTIPLE tree searches here, or is this loop just used to expand a single tree?
        #  Based on what I know I would assume the latter, but why does the for loop set the node to root each time? Are we making the tree branch by branch rather than layer by layer?
        #  But what about branches that have the same starting nodes?
        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]  # So is this to hold a branch of the tree? Or an entire tree?
            current_tree_depth = 0

            # Run the rest of the tree search. Not actually in this while loop, though, this while loop is just used to... what?
            # I think this is used to create a single branch? Not the rest of the tree search? But wouldn't that need the dynamics function?
            # Wait, or is this used to skip through the nodes that we've already created in a branch and skip to the leaf of the branch so we can expand it? Is that it?
            while node.expanded():
                current_tree_depth += 1  # Increment tree depth
                action, node = self.select_child(node, min_max_stats)  # Select best action and determine the next node from the current node based on the min_max_stats.
                search_path.append(node)

                # Players play turn by turn. But wait, wouldn't this if-else only work for two player games? What about 3+ players?
                # Oh, wait, below it says "More than two player mode not implemented." Ok. Whatever, our game is 1 player anyway
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            # Here's the REALLY cool part:
            # Inside the search tree we use the dynamics function to obtain the next hidden state given an action and the previous hidden state.
            parent = search_path[-2]
            # model.recurrent_inference is where we call the dynamics function to obtain each hidden state from its previous state, including the hidden state from the representation function from the initial observation.
            value, reward, policy_logits, hidden_state = model.recurrent_inference(parent.hidden_state, torch.tensor([[action]]).to(parent.hidden_state.device))
            value = models.support_to_scalar(value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            node.expand(self.config.action_space, virtual_to_play, reward, policy_logits, hidden_state)  # create a new node from the previous one?

            self.backpropagate(search_path, value, virtual_to_play, min_max_stats)  # Backpropagate to improve the network or tree or something.

            max_tree_depth = max(max_tree_depth, current_tree_depth)  # The depth of the biggest tree we've made here.

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value
        }
        return root, extra_info  # FIXME: So why are we returning the root? Are we... using the root value to determine the next move or something?

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score, aka the best action.
        """
        max_ucb = max(self.ucb_score(node, child, min_max_stats) for action, child in node.children.items())  # Find the action with the highest ucb score of the node's children.
        # Select an action randomly from a list of children with the highest ucb score. So wait, wouldn't that list only have one item? Unless there are two children with the max score.
        action = numpy.random.choice([action for action, child in node.children.items() if self.ucb_score(node, child, min_max_stats) == max_ucb])
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        """
        Find the ucb score for a node based on its value, plus an exploration bonus based on the prior. I'm not sure what a prior is.
        """
        # Basically this is all voodoo, so unfortunately I don't have much to say about it.
        # The score seems to mostly be based on the node's visit count, so where do the neural nets come into play?
        # OH! The neural nets come into play in backpropagate(), as backpropagate() uses the reward and value from the dynamics function to improve the min_max_stats! Great!
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q
            # Is this some kind of Q-learning thing?
            value_score = min_max_stats.normalize(
                child.reward
                + self.config.discount
                * (child.value() if len(self.config.players) == 1 else -child.value())
            )
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree to the root.
        So is a simulation an exploration of a tree branch? So would num_simulations be the number of branches in a tree?
        In that case, it makes perfect sense why you can still get such great results with MuZero with very few simulations, because the neural nets figure out which
        branches are the most promising before we even get started, so we don't really need that many, as the ones the neural nets come up with will already be really good.
        """
        if len(self.config.players) == 1:
            # For each node in the search path, update the min_max_stats with the reward and value from the dynamics function.
            for node in reversed(search_path):
                node.value_sum += value  # Add value from dynamics function to value_sum
                node.visit_count += 1  # Increment node visit count.
                min_max_stats.update(node.reward + self.config.discount * node.value())  # Update the min_max_stats with the reward and value from the dynamics function! So this is where the NNs come into play!
                # Adjust value with reward and discount values.
                value = node.reward + self.config.discount * value
        elif len(self.config.players) == 2:
            for node in reversed(search_path):
                node.value_sum += value if node.to_play == to_play else -value  # Add -value for opponent's value.
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * -node.value())
                value = (-node.reward if node.to_play == to_play else node.reward) + self.config.discount * value
        else:
            raise NotImplementedError("More than two player mode not implemented.")


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    # Return whether or not the node is expanded, i.e. it has children.
    def expanded(self):
        return len(self.children) > 0

    # Return the TRUE value of node, which is actually the value_sum / visit_count. So the less visited the node is, and the higher its value, means it's overall value is higher, which leads to muzero exploring promising, yet less tested nodes. Cool!
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    # Here's a lot of the meat of how the MCTS is created, and this is where the information from the neural nets is used the most.
    def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        """
        Expand a node, i.e. create new children for it, using the value, reward, and policy prediction from the neural network! Cool, so this is also where the neural nets come into play!

        :param actions:         Legal actions
        :param to_play:         Player number
        :param reward:          The predicted reward from either the initial_inference (which uses some weird voodoo I don't understand) or the recurrent_inference using the dynamics function.
        :param policy_logits:   The policy/action to take.
        :param hidden_state:    The hidden state from either the representation function or the dynamics function.
        :return:                void, no return
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        policy_values = torch.softmax(torch.tensor([policy_logits[0][a] for a in actions]), dim=0).tolist()  # Find values for each action using the policy,
        policy = {a: policy_values[i] for i, a in enumerate(actions)}  # use values to select best actions,
        for action, p in policy.items():
            self.children[action] = Node(p)  # and use make a node from each of the best actions! Awesome!

    # At the start of each search, add noise to the prior of the root to encourage the search to explore new actions.
    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to encourage the search to explore new actions.
        The way we use dirichlet noise and why is explained well in this video: https://youtu.be/L0A86LmH7Yw?t=697
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class GameHistory:
    """
    Store only useful information of a self-play game. FIXME: Where does it store it? RAM? I hope it doesn't store it there permanently...
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For Prioritized Replay
        self.priorities = None
        self.game_priority = None

    # Store values of roots of each tree to root_values.
    def store_search_statistics(self, root, action_space):
        # Turn visit count from root into a policy, aka a set of best actions I think.
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            # Append child_visits... with what and why?
            self.child_visits.append(
                [
                    root.children[a].visit_count / sum_visits
                    if a in root.children
                    else 0
                    for a in action_space
                ]
            )
            self.root_values.append(root.value())
        else:
            self.root_values.append(None)

    # Get the previous actions and observations to add to the current observation.
    def get_stacked_observations(self, index, num_stacked_observations):
        """
        Generate a new observation with the observation at the index position and num_stacked_observations past observations and actions stacked.

        :param index:                       Index from which we start stacking the observations.
        :param num_stacked_observations:    Number of past observations and actions stacked.
        :return:                            stacked_observations
        """
        # Convert index to positive value
        index = index % len(self.observation_history)

        stacked_observations = self.observation_history[index].copy()  # Copy observations from index in observation_history.
        # For all past observations from the index to the index - num_stacked_observations, get previous observations
        for past_observation_index in reversed(range(index - num_stacked_observations, index)):
            # Not exactly sure what this voodoo is for.
            if 0 <= past_observation_index:
                previous_observation = numpy.concatenate(  # numpy.concatenate concatenates a tuple of arrays, hence the seemingly redundant double parentheses.
                    (
                        self.observation_history[past_observation_index],
                        [numpy.ones_like(stacked_observations[0]) * self.action_history[past_observation_index + 1]]
                    )
                )
            else:
                previous_observation = numpy.concatenate(
                    (
                        numpy.zeros_like(self.observation_history[index]),
                        [numpy.zeros_like(stacked_observations[0])]
                    )
                )

            stacked_observations = numpy.concatenate((stacked_observations, previous_observation))

        return stacked_observations
            ### Alright, we'll figure out exactly what this stuff does later, for now, back to play_game()

# The word "stats" makes it sound like a big set of values or a probability distribution or something, but no, it's literally two float values: minimum and maximum.
class MinMaxStats:
    """
    A class that holds the min-max values of the tree. Literally two float values: minimum and maximum.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values no maximum will be greater than the default infinity that the min is set to. Clever.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

    ### Wow, despite all my comments, this file is STILL slightly shorter than the original because I didn't expand every parentheses in sight.
    ### Anyway, we're done with this whole thing! Awesome, let's head back over to muzero.py now!