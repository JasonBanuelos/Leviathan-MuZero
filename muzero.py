# Official Python Modules
import copy
import importlib
import math
import os
import pickle
import sys
import time
from glob import glob

# 3d party Packages
import nevergrad
import numpy
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

# Files & classes that we've created for this project
import models
import replay_buffer
import self_play
import shared_storage
import trainer


### If you're a total noob to python, go down to the program's main function first at the bottom of the file, which is declared like this: if __name__ == "__main__":
class MuZero:
    """
    Main class to manage Leviathan's MuZero brain.

    Arguments:
        game_name (str): Name of the game module, it should match the name of a .py file in the games folder.
        config (dict or MuZeroConfig, optional): Override the default config of the game.
        split_resources_in (int, optional): Split the GPU usage when using concurent muzero instances. I think it's the percent of the GPU that's split?

    Example:
        >>> muzero = MuZero("cartpole")
        >>> muzero.train()
        >>> muzero.test(render=True)
    """

    # __init__ is a python class's constructor.
    def __init__(self, game_name, config=None, split_resources_in=1):
        # Load the game and its config for muzero
        try:
            game_module = importlib.import_module("games." + game_name)  # Load the module from a game file in the games folder which contains the game.
            self.Game = game_module.Game  # Both .Game and .MuZeroConfig() are in the game file from which the game is being loaded.
            self.config = game_module.MuZeroConfig()
        # If the game specified is not defined in the games folder, raise an error and alert the user. Ex: "python muzero.py destoryallhumans" would raise an error.
        except ModuleNotFoundError as err:
            print(f'{game_name} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.')  # He used '' instead of "" so he could put cartpole in quotes. Clever.
            raise err

        # If there is no config input (i.e. config=None), then this whole thing is bypassed.
        if config:
            if type(config) is dict:
                for param, value in config.items():
                    setattr(self.config, param, value) # Sets the named attribute on the given object to the specified value, e.g. setattr(x, 'y', v) is equivalent to ``x.y = v''
            else:
                self.config = config

        # Set torch & numpy's RNG seed to whatever you specified in the game file.
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # These next few if statements make sure GPUs are used properly.
        # Raise an error if max_num_gpus = 0 but GPU requested by config file
        if self.config.max_num_gpus == 0 and (self.config.selfplay_on_gpu or self.config.train_on_gpu or self.config.reanalyse_on_gpu):  # I honestly think it's a lot less confusing if you keep all the if statement stuff on one line, rather than spreading it out like Duvaud does.
            raise ValueError("Inconsistent MuZeroConfig: max_num_gpus = 0 but GPU requested by selfplay_on_gpu or train_on_gpu or reanalyse_on_gpu.")
        # If using gpus, set total_gpus equal to however many gpus you're using.
        if self.config.selfplay_on_gpu or self.config.train_on_gpu or self.config.reanalyse_on_gpu:
            # if self.config.max_num_gpus is not None, then total_gpus = self.config.max_num_gpus
            # else, total_gpus = torch.cuda.device_count()
            # I think this way of writing it looks kind of confusing, but whatever. At least it's short. Kind of.
            total_gpus = (
                self.config.max_num_gpus
                if self.config.max_num_gpus is not None
                else torch.cuda.device_count()
            )
        else:
            total_gpus = 0
        self.num_gpus = total_gpus / split_resources_in  # Uhh, but if split_resources_in = 0, then wouldn't we be dividing by zero?
        # If using more than 1 gpu, round down to make sure number of gpus used is an integer.
        if 1 < self.num_gpus:
            self.num_gpus = math.floor(self.num_gpus)

        # Start up ray
        ray.init(num_gpus=total_gpus, ignore_reinit_error=True)

        # So as far as I can tell, the way this thing works is that it initializes "workers" which train MuZero using "checkpoints". Checkpoints are essentially MuZero models that it's trained, which you can access and train more or play against later.
        # Checkpoints I think are what allow you to stop MuZero in the middle of training and save its progress so you don't have to be running it for multiple days straight.
        # It then loads these workers into a replay buffer i think, where it stores played-out games (called "trajectories" in deep RL) and learns from them.
        # Check this video out to see MuZero's two main components: one that plays and one that learns. https://youtu.be/L0A86LmH7Yw?t=1319 ...But why can't it learn AS it plays?
        # Here we initialize the checkpoint and replay buffer used to initialize workers
        # This part is actually REALLY important. This defines all the data that is stored by shared_storage.
        self.checkpoint = {
            "weights": None,
            "optimizer_state": None,
            "total_reward": 0,
            "muzero_reward": 0,
            "opponent_reward": 0,
            "episode_length": 0,
            "mean_value": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "num_played_games": 0,
            "num_played_steps": 0,
            "num_reanalysed_games": 0,
            "terminate": False,
        }
        self.replay_buffer = {}

        cpu_actor = CPUActor.remote()  # CPUActor is a class defined below. I implemented it right after writing this line, just to get a hold of how it works.
        cpu_weights = cpu_actor.get_initial_weights.remote(self.config)  # Get the actors weights according to the game's config or something.
        self.checkpoint["weights"], self.summary = copy.deepcopy(ray.get(cpu_weights))  # Get both the checkpoint weights and the summary. The summary is used save the model representation.
        # So is there no GPU actor? that seems limiting if true.

        # Initialize all types of workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    ### After I finished the above constructor, I tried to run the program and it gave me the error "ModuleNotFoundError: No module named 'models'", so I decided to start the models file. You might want to go there next to see how it works before you finish reading this file.

    def train(self, log_in_tensorboard=True):
        """
        Pretty self explanatory, spawn ray workers and start training.

        Args:
            log_in_tensorboard (bool): Whether or not to start a testing worker and log its performance in TensorBoard. or something
        """
        # Whether or not to save the checkpoint we're training. Since log_in_tensorboard is true by default, this will run by default.
        # This creates a "results" directory if none exists and will save checkpoints to it. Cool!
        if log_in_tensorboard or self.config.save_model:  # save_model is a bool in the game config.
            os.makedirs(self.config.results_path, exist_ok=True)  # Make directory at results_path where checkpoints will be saved. results_path, defined in the game file, automatically creates a "results" directory, a directory for the specific game's results, and a directory for the checkpoint. Take a look at it in any of Duvaud's game files, it's actually really cool!

        # Figure out how we're going to use our GPU(s)
        if 0 < self.num_gpus:
            num_gpus_per_worker = self.num_gpus / (  # Another one of those ugly expressions that you have to spread out over multiple lines.
                # This boolean expression determines num_gpus_per_worker https://www.allaboutcircuits.com/textbook/digital/chpt-7/boolean-arithmetic/
                # You know what's hilarious? I saw this expression and was totally confused because I completely forgot that you could add and multiply booleans, despite the fact that I learned about boolean arithmetic just last semester in a college course that I got an A in. I can't wait to graduate...
                # So this can result in using a fraction of a GPU per worker depending on the result of this expression, which I guess make sense, but what if all the bools are false? Wouldn't we get a divide by 0?
                # Oh wait! So looking back at the init, if train_on_gpu, selfplay_on_gpu, and reanalyse_on_gpu are all false, then num_gpus = 0, so this block of code won't even run, so there's no way this expression will result in 0. ...Right?
                # UNLESS use_last_model_value = False while reanalyse_on_gpu = True and both train_on_gpu and selfplay_on_gpu are False, that could result in a divide by 0.
                # However, it is unlikely that use_last_model_value would be False while reanalyse_on_gpu would be true, as use_last_model_value is necessary for reanalyze I think.
                # FIXME: So it's possible, though unlikely, that this expression would result in a divide by zero, but that would have to be because someone who doesn't understand the algo tried to reanalyze on GPU with reanalyze off. I'll fix it later. Perhaps something like 'self.use_last_model_value, self.reanalyse_on_gpu = True' in one line in the game file
                self.config.train_on_gpu
                + self.config.num_workers * self.config.selfplay_on_gpu
                + log_in_tensorboard * self.config.selfplay_on_gpu
                + self.config.use_last_model_value * self.config.reanalyse_on_gpu
            )
            # Similar to what we did in __init__
            if 1 < num_gpus_per_worker:
                num_gpus_per_worker = math.floor(num_gpus_per_worker)
        else:
            num_gpus_per_worker = 0
        # Just tried running this thing, and I was actually able to use the train option! It didn't really do anything, and it went right back to the option menu immediately after running, but it didn't give me an error! It even created a "results" directory and saved a result! Cool!

        # Initialize training workers.
        self.training_worker = trainer.Trainer.options(  # .Trainer is defined in trainer.py, .options is defined in ray.
            num_cpus=0,
            num_gpus=num_gpus_per_worker if self.config.train_on_gpu else 0,
        ).remote(self.checkpoint, self.config)

        ### Since I had no trainer.py yet, I got ModuleNotFoundError: No module named 'trainer' after running what I had so far, so I decided to code that next. Head on over there!

        # Initialize storage workers
        self.shared_storage_worker = shared_storage.SharedStorage.remote(self.checkpoint, self.config)
        self.shared_storage_worker.set_info.remote("terminate", False)

        # Initialize replay buffer workers.
        self.replay_buffer_worker = replay_buffer.ReplayBuffer.remote(self.checkpoint, self.replay_buffer, self.config)

        ### Since I had no replay_buffer.py yet, I got ModuleNotFoundError: No module named 'replay_buffer' after running this. I decided to go code the init for that next, so head on over to replay_buffer.py!

        # Part of Reanalyze, use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze). We probably won't reanalyze due to the nature of our environment.
        if self.config.use_last_model_value:
            self.reanalyse_worker = replay_buffer.Reanalyze.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.reanalyse_on_gpu else 0,  # I know reanalyze is spelled wrong here, but it would just be a pain to correct all the misspellings, so for now I will leave some of them as they are.
            ).remote(self.checkpoint, self.config)

        # Initialize a list of self play workers for seed in range(self.config.num_workers)
        self.self_play_workers = [
            self_play.SelfPlay.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
            ).remote(self.checkpoint, self.Game, self.config, self.config.seed + seed)
            for seed in range(self.config.num_workers)
        ]

        ### No self_play.py yet, so got ModuleNotFoundError: No module named 'self_play'. Decided to create the init for that, too, so head on over there!

        # Launch workers
        [
            self_play_worker.continuous_self_play.remote(self.shared_storage_worker, self.replay_buffer_worker)  # I don't know why this is in a list like this, but whatever.
            for self_play_worker in self.self_play_workers
        ]
        self.training_worker.continuous_update_weights.remote(self.replay_buffer_worker, self.shared_storage_worker)
        if self.config.use_last_model_value:
            self.reanalyse_worker.reanalyse.remote(self.replay_buffer_worker, self.shared_storage_worker)

        if log_in_tensorboard:
            self.logging_loop(num_gpus_per_worker if self.config.selfplay_on_gpu else 0)

        ### Alright, we're done with the train function! Kind of! Okay, we got an error telling us: AttributeError: 'ActorHandle' object has no attribute 'continuous_self_play'
        ### So let's go over to self_play.py and implement that!

        ### Okay, after implementing self_play.py, I tried to run it again, but this time I got an error telling me the Reanalyze class has no reanalyze function,
        ### and I realized that I needed to finish the Reanalyze class.
        ### But I also realized that there were a bunch of functions and classes and things that muzero.py relied on in both replay_buffer.py and trainer.py,
        ### so instead of jumping around and patching holes here and there, I'm just going to finish ALL of replay_buffer.py and then ALL of trainer.py,
        ### So let's head on over to replay_buffer.py, under the init, and finish everything up there. We're a little over halfway done with the whole thing, I think!

        ### Okay, we've finally finished both replay_buffer.py and trainer.py! But after running the code, I still got an error:
        ### AttributeError: 'MuZero' object has no attribute 'logging_loop'
        ### Whoops. Forgot about that. Fortunately, though, we don't have to go to another file to implement that, so keep reading and let's get it done!

    def logging_loop(self, num_gpus):
        """
        Use TensorBoard to keep track of training performance.
        """
        # Launch the test worker to get performance metrics
        self.test_worker = self_play.SelfPlay.options(num_cpus=0, num_gpus=num_gpus).remote(  # Sorry I split this up, but it was too long for one line even for me.
            self.checkpoint, self.Game, self.config, self.config.seed + self.config.num_workers
        )
        self.test_worker.continuous_self_play.remote(self.shared_storage_worker, None, True)

        # Write everything in TensorBoard
        writer = SummaryWriter(self.config.results_path)  # Creates a `SummaryWriter` that will write out events and summaries to the event file

        print("\nTraining...\nRun 'tensorboard --logdir ./results' and go to http://localhost:6006/ to see in real time the training performance.\n")

        # Save hyperparameters to TensorBoard
        hp_table = [
            f"| {key} | {value} |" for key, value in self.config.__dict__.items()
        ]
        writer.add_text(
            "Hyperparameters",
            "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table)
        )
        # Save model representation
        writer.add_text("Model summary", self.summary)
        # Loop for updating the training performance
        counter = 0  # Initialize counter for... something? So we can keep track of how many times the upcoming loop has iterated?
        keys = [
            "total_reward",
            "muzero_reward",
            "opponent_reward",
            "episode_length",
            "mean_value",
            "training_step",
            "lr",
            "total_loss",
            "value_loss",
            "reward_loss",
            "policy_loss",
            "num_played_games",
            "num_played_steps",
            "num_reanalysed_games",
        ]
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))  # Get info to determine training step.
        # For each training step, save a bunch of data to TensorBoard. Save rewards, worker info, and losses.
        try:
            while info["training_step"] < self.config.training_steps:
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
                writer.add_scalar("1.Total_reward/1.Total_reward", info["total_reward"], counter)
                writer.add_scalar("1.Total_reward/2.Mean_value", info["mean_value"], counter)
                writer.add_scalar("1.Total_reward/3.Episode_length", info["episode_length"], counter)
                writer.add_scalar("1.Total_reward/4.MuZero_reward", info["muzero_reward"], counter)
                writer.add_scalar("1.Total_reward/5.Opponent_reward", info["opponent_reward"], counter)  # In his code, this scalar was spread over three lines, rather than one, like the rest. Once again, code generator?
                writer.add_scalar("2.Workers/1.Self_played_games", info["num_played_games"], counter)
                writer.add_scalar("2.Workers/2.Training_steps", info["training_step"], counter)  # Or perhaps sloppy code style? He typically had a comma after the 'counter' argument, but this time he didn't.
                writer.add_scalar("2.Workers/3.Self_played_steps", info["num_played_steps"], counter)
                writer.add_scalar("2.Workers/4.Reanalysed_games", info["num_reanalysed_games"], counter)
                writer.add_scalar("2.Workers/5.Training_steps_per_self_played_step_ratio", info["training_step"] / max(1, info["num_played_steps"]), counter)
                writer.add_scalar("2.Workers/6.Learning_rate", info["lr"], counter)
                writer.add_scalar("3.Loss/1.Total_weighted_loss", info["total_loss"], counter)
                writer.add_scalar("3.Loss/Value_loss", info["value_loss"], counter)
                writer.add_scalar("3.Loss/Reward_loss", info["reward_loss"], counter)
                writer.add_scalar("3.Loss/Policy_loss", info["policy_loss"], counter)
                print(
                    f'Last test reward: {info["total_reward"]:.2f}. Training step: {info["training_step"]}/{self.config.training_steps}. Played games: {info["num_played_games"]}. Loss: {info["total_loss"]:.2f}',
                    end="\r",
                )
                counter += 1
                time.sleep(0.5)
        # So wait, this is actually pretty big! KeyboardInterrupt means if we hit ctrl+C here, we pass AND SKIP THE REST OF THE FUNCTION.
        # This is important because hitting ctrl+C here SKIPS THE PART WHERE WE SAVE THE REPLAY BUFFER TO DISK! That means we'll lose all our progress, right?!
        # This may be why I lost all my progress when ctrl+C-ing in the middle of training for atari breakout, to try to save some progress before the thing crashed.
        # So if we move the save_model part INTO THE EXCEPTION, would it still save the model if we hit ctrl+C?
        # It also doesn't properly terminate the workers, as is.
        except KeyboardInterrupt:
            pass

        self.terminate_workers()

        if self.config.save_model:
            # Persist(?) replay buffer to disk
            print("\n\nPersisting replay buffer games to disk...")
            # Serialize the replay buffer data in the curly braces and save it to the file we open.
            pickle.dump(  # pickle serializes, or "pickles", a python object
                {
                    "buffer": self.replay_buffer,
                    "num_played_games": self.checkpoint["num_played_games"],
                    "num_played_steps": self.checkpoint["num_played_steps"],
                    "num_reanalysed_games": self.checkpoint["num_reanalysed_games"],
                },
                open(os.path.join(self.config.results_path, "replay_buffer.pkl"), "wb")
            )

    def terminate_workers(self):
        """
        Softly terminate the running tasks and garbage collect the workers.
        """
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info.remote("terminate", True)
            self.checkpoint = ray.get(self.shared_storage_worker.get_checkpoint.remote())
        if self.replay_buffer_worker:
            self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

        print("\nShutting down workers...")

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

        ### Alright, I think we've got everything we need to train this thing! We'll implement test() and all the other stuff later, but for now lets see if we can start
        ### training it to play a game.
        ### Okay, got some errors, but I think they were because of minor mistakes I made in copying the code. Also, forgot to implement scalar_to_support in models.py, so I went over and did that.

        ### IT'S WORKING! IT'S WORKING!!! I'll probably comb over the code one last time just to be sure I didn't mess anything up, but it's actually training! Awesome!
        ### Actually, before I comb over all the code again, let's finish the rest of muzero.py and then implement resnet in models.py. That way we can let it train for a while and test it to make sure it actually works. This is so awesome!

    def test(self, render=True, opponent=None, muzero_player=None, num_tests=1, num_gpus=0):
        """
        Test the model by playing a game in a dedicated thread.

        :param render (bool):       To display or not the environment. Defaults to True.
        :param opponent (str):      The opponent for muzero to play against, "self" for self-play, "human" for playing against MuZero and "random" for a random agent,
                                    None will use the opponent in the config. Defaults to None.
        :param muzero_player (int): Player number of MuZero in case of multiplayer games, None let MuZero play all players turn by turn,
                                    None will use muzero_player in the config. Defaults to None.
        :param num_tests (int):     Number of games to average. Defaults to 1.
        :param num_gpus (int):      Number of GPUs to use, 0 forces to use the CPU. Defaults to 0.
        :return: result (list):     Mean of rewards earned from a self play worker playing the game.
        """
        opponent = opponent if opponent else self.config.opponent
        muzero_player = muzero_player if muzero_player else self.config.muzero_player
        self_play_worker = self_play.SelfPlay.options(num_cpus=0, num_gpus=num_gpus).remote(self.checkpoint, self.Game, self.config, numpy.random.randint(10000))
        results = []  # Initialize a list to store the results of self play
        # Self play the game for the number of games/tests specified.
        for i in range(num_tests):
            print(f"Testing {i+1}/{num_tests}")
            results.append(ray.get(self_play_worker.play_game.remote(0, 0, render, opponent, muzero_player)))
        self_play_worker.close_game.remote()

        # If singleplayer, find mean of rewards, if multiplayer, only find the mean of muzero's rewards.
        if len(self.config.players) == 1:
            result = numpy.mean([sum(history.reward_history) for history in results])
        else:
            result = numpy.mean(
                [
                    sum(
                        reward for i, reward in enumerate(history.reward_history)
                        if history.to_play_history[i - 1] == muzero_player
                    )
                    for history in results
                ]
            )
        return result  # Return mean of muzero's rewards

    # Used by load_model_menu to load the model from disk.
    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        """
        Load a model and/or a saved replay buffer.

        :param checkpoint_path:    (str) Path to model.checkpoint or model.weights.
        :param replay_buffer_path: (str) Path to replay_buffer.pkl
        :return:
        """
        # Load checkpoint from disk.
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                self.checkpoint = torch.load(checkpoint_path)
                print(f"\nUsing checkpoint from {checkpoint_path}")
            else:
                print(f"\nThere is no model saved in {checkpoint_path}.")

        # Load replay buffer from disk.
        if replay_buffer_path:
            if os.path.exists(replay_buffer_path):
                with open(replay_buffer_path, "rb") as f:
                    replay_buffer_info = pickle.load(f)  # Deserialize the replay buffer data from disk with pickle.
                self.replay_buffer = replay_buffer_info["buffer"]
                self.checkpoint["num_played_steps"] = replay_buffer_info["num_played_steps"]
                self.checkpoint["num_played_games"] = replay_buffer_info["num_played_games"]
                self.checkpoint["num_reanalysed_games"] = replay_buffer_info["num_reanalysed_games"]

                print(f"\nInitializing replay buffer with {replay_buffer_path}")
            else:
                print(f"Warning: Replay buffer path '{replay_buffer_path}' doesn't exist.  Using empty buffer.")
                self.checkpoint["training_step"] = 0
                self.checkpoint["num_played_steps"] = 0
                self.checkpoint["num_played_games"] = 0
                self.checkpoint["num_reanalysed_games"] = 0

        ### This function is used by load_model_menu() to load the models, so let's go down and implement that, then come back up and continue.

    # FIXME: I've tried to use this a few times, and every time, the matplotlib plots were broken, I'm not sure why.
    def diagnose_model(self, horizon):
        """
        Play a game only with the learned model then play the same trajectory in the real environment and display information.
        Meant to make sure the representation function's model is accurately representing the real environment.

        :param horizon: (int) Number of timesteps for which we collect information.
        :return: Void
        """
        game = self.Game(self.config.seed)
        obs = game.reset()
        dm = diagnose_model.DiagnoseModel(self.checkpoint, self.config)
        dm.compare_virtual_with_real_trajectories(obs, game, horizon)
        input("Press enter to close all plots")
        dm.close_all()

        ### Alright, we'll implement diagnose_model.py and hyperparameter_search() later, for now, lets train this thing and test it to make sure it's really learning.
        ### YES! IT WORKS! It trains to completion, loads the model, and kicks cartpole's ass! Awesome!
        ### Now all that's left to do is implement diagnose_model.py, hyperparameter_search(), and resnet in models.py.
        ### I'd like to be able to use resnet, and I don't really care about hyperparameter_search() or diagnose_model.py since I'm not really sure what those do
        ### and since diagnose_model() didn't really seem to work, so let's implement resnet first. Off to model.py!


# Defines a CPU worker.
@ray.remote(num_cpus=0, num_gpus=0) # ray.remote defines a remote function OR an actor class, so I think we're using the latter.
class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    # Uhh, what? DataParallel is what "Implements data parallelism at the module level", but why would we want it to stay on CPU if there's a GPU?
    def __init__(self):
        pass

    def get_initial_weights(self, config):
        model = models.MuZeroNetwork(config)  # Set the worker's model as the muzero neural network defined in the game's config. MuZeroNetwork() is defined in models.py
        weights = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weights, summary


# Used in main to present the models we can load.
def load_model_menu(muzero, game_name):
    # Make a list of models the user can load from the results folder. Also include the options to specify paths manually
    options = ["Specify paths manually"] + sorted(glob(f"results/{game_name}/*/"))  # The glob module finds all the pathnames matching a specified pattern used by the Unix shell, although results are returned in arbitrary order, which is why we sorted it. So it's a bit like RegEx?
    options.reverse()
    print()
    for i in range(len(options)):
        print(f"{i}. {options[i]}")

    choice = input("Enter a number to choose a model to load: ")
    valid_inputs = [str(i) for i in range(len(options))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")
    choice = int(choice)

    # Either allow user to enter path manually or load the model which the user has selected.
    if choice == (len(options) - 1):
        checkpoint_path = input("Enter a path to the model.checkpoint, or hit ENTER if none: ")
        while checkpoint_path and not os.path.isfile(checkpoint_path):
            checkpoint_path = input("Invalid checkpoint path. Try again: ")
        replay_buffer_path = input("Enter a path to the replay_buffer.pkl, or ENTER if none: ")
        while replay_buffer_path and not os.path.isfile(replay_buffer_path):
            replay_buffer_path = input("Invalid replay buffer path. Try again: ")
    else:
        checkpoint_path = f"{options[choice]}model.checkpoint"
        replay_buffer_path = f"{options[choice]}replay_buffer.pkl"

    muzero.load_model(checkpoint_path=checkpoint_path, replay_buffer_path=replay_buffer_path)


### This is how you declare main in python. This is the part of the file that actually runs the program. Kind of weird that the most important part of the code is at the bottom of the file, but I guess that's so the functions & classes and stuff that the program uses are compiled before the program starts.
if __name__ == "__main__":
    # If you define the game you want it to play, then it skips the menu and gets straight to training itself to play whatever you want.
    # So if you enter something in the terminal like "python muzero.py cartpole" then it trains it to play cartpole.
    if len(sys.argv) == 2:
        print("Training MuZero to play " + sys.argv[1] + "!")
        muzero = MuZero(sys.argv[1])
        muzero.train()
    else:
        # Ask the user what game it wants muzero to play before initializing it.
        print("\nWelcome to MuZero! Here's a list of games:")
        # Make a list of the games to be printed.
        games = [
            filename[:-3]  # Trim the extensions from the filenames. I have no idea why this comes before the next statement and not after. I learned the hard way in skool that order matters in code.
            # Get all the file names in the games directory in the directory where this file is, and put them in a sorted list.
            for filename in sorted(
                os.listdir(os.path.dirname(os.path.realpath(__file__)) + "/games")
            )
            # Add the file name to the list of games if it's a .py and if it's not the abstract game file. I don't know why this if statement is down here, though.
            if filename.endswith(".py") and filename != "abstract_game.py"
        ]
        # Print the list of games.
        for i in range(len(games)):
            print(f"{i}. {games[i]}")  # This may seem a little convoluted, but this allows us to print the game and the game's number. The f at the front is python's formatting thing. You could add +1 to {i} to make the list start from 1 rather than 0, but it makes the code more convoluted later, so its best to just keep it starting at 0.

        choice = input("Enter a number to choose the game: ")  # BTW, there's no need to declare this variable as a string in python. Python figures out typing on its own.
        valid_inputs = [str(i) for i in range(len(games))]  # A one-line declaration of a list of the game numbers. Pretty cool, huh?
        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")

        # Initialize MuZero
        choice = int(choice)  # Easy way to convert a string to an int.
        game_name = games[choice]
        muzero = MuZero(game_name)

        # Now it's showtime
        while True:
            # Ask what you want MuZero to do with the game you've selected.
            options = [
                "Train",
                "Load pretrained model",
                "Diagnose model",
                "Render some self play games",
                "Play against MuZero",
                "Test the game manually",
                "Hyperparameter search",
                "Exit",
            ]
            print()  # Print a newline, I guess.
            for i in range(len(options)):
                print(f"{i}. {options[i]}")  # Similar to printing the games & game numbers above, but for the options.

            choice = input("Enter a number to choose an action: ")
            valid_inputs = [str(i) for i in range(len(options))]
            while choice not in valid_inputs:
                choice = input("Invalid input, enter a number listed above: ")
            choice = int(choice)
            # No switch statement in python. Eh, those things are kind of redundant anyway.
            if choice == 0:
                muzero.train()
            elif choice == 1:
                load_model_menu(muzero, game_name)  # Static function defined above in this file.
            elif choice == 2:
                muzero.diagnose_model(30)  # Check the muzero class for all the definitions of these functions.
            elif choice == 3:
                muzero.test(render=True, opponent="self", muzero_player=None)  # Is it necessary to have muzero_player=None as a third argument here? That argument defaults to None anyway
            elif choice == 4:
                muzero.test(render=True, opponent="human", muzero_player=0)
            elif choice == 5:
                # Define the environment (game) you're testing. Reset it and render it.
                env = muzero.Game()  # In Pycharm, ctrl+click Game() to find out how the game is defined.
                env.reset()  # Every function that the environment uses is defined in the game file of that game.
                env.render()

                # Start playing the game
                done = False
                while not done:
                    action = env.human_to_action()  # Get your action.
                    observation, reward, done = env.step(action)  # Input the action into the environment. Get the observation, reward, and whether or not you're done.
                    print(f"\nAction: {env.action_to_string(action)}\nReward: {reward}")  # Print out the action you took and the reward you got for it.
                    env.render()  # Render the environment after you take the action.
            elif choice == 6:
                # Define here the parameters to tune
                # Parametrization documentation: https://facebookresearch.github.io/nevergrad/parametrization.html
                # "The aim of parametrization is to specify what are the parameters that the optimization should be performed upon."
                # "the parameter modules (accessed by the shortcut nevergrad.p) provide classes that should be used to specify each parameter"  What?
                muzero.terminate_workers()
                del muzero  # We're creating a whole new muzero from scratch for this.
                # I'll be honest, as of writing this, I have no clue what all this is for. Look at the hyperparameter_search function defined above, I guess.
                budget = 20
                parallel_experiments = 2
                lr_init = nevergrad.p.Log(lower=0.0001, upper=0.1)  # nevergrad.p.Log() is a "Parameter representing a positive variable, mutated by Gaussian mutation in log-scale." What?
                discount = nevergrad.p.Log(lower=0.95, upper=0.9999)  # May need to look at the muzero paper to find out what these parameters are supposed to be.
                parameterization = nevergrad.p.Dict(lr_init=lr_init, discount=discount) # "Dictionary-valued parameter. This Parameter can contain other Parameters, its value is a dict, with keys the ones provided as input, and corresponding values are either directly the provided values if they are not Parameter instances, or the value of those Parameters."
                best_hyperparameters = hyperparameter_search(game_name, parameterization, budget, parallel_experiments, 20)
                muzero = MuZero(game_name, best_hyperparameters) # FIXME: So does this replace the MuZero we use to play this game permanently forever? Or just for this one time we run the program?
            else:
                print("ᛊieg der ᛊonne")  # Hilariously, his "Done" print statement doesn't work because it's after the break. Like I said earlier, I learned the hard way in skool that order matters in code.
                break  ### Alright, once you get how this works, go back up to MuZero class

    ray.shutdown()