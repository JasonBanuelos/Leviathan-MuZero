![supported platforms](https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20%7C%20Windows%20(Jupyter)-929292)
![supported python versions](https://img.shields.io/badge/python-%3E%3D%203.6-306998)
![dependencies status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)
[![style black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![license MIT](https://img.shields.io/badge/licence-MIT-green)

# Leviathan MuZero

Over the course of two or three weeks, I copied [Werner Duvaud's implementation of MuZero](https://github.com/werner-duvaud/muzero-general) line by line, by hand, to pick apart how it works. I had almost no experience with Python before I started copying this code, other than a few beginner Python and PyTorch tutorials that I had gone through. I wrote comments all over this implementation, explaining how it works to the best of my ability. Hopefully the comments are clear enough for other beginners to follow along and read about how this thing works, if they'd like. I also wrote comments on the parts that I didn't understand so I could come back to them later for further study. There are still a few features that I haven't copied yet that I intend to return to later, as they aren't particularly useful to me at the moment. For more information, take a gander at Duvaud's repository on GitHub.

I'll update this readme with more information on what I plan to do with this implementation later. [Here's my latest description of the project.](https://docs.google.com/document/d/19QZMQqhQWLLvkfISsEOnBnU440sNAXr3/edit?usp=sharing&ouid=113065030893762115002&rtpof=true&sd=true)


## FIXME

* [ ] The replay buffer slowly fills up memory, causing the program to break at around 60000 training steps. When it breaks, we lose all training progress. Probably some kind of memory leak?
      Alright, I think I figured out a solution? I changed the replay buffer size in breakout to only 32, and the replay buffer never filled up more than about 2082 MiBs of ram, effectively halting its death by attrition.
* [ ] When trying to load and traim a model that's been through like 30000 training steps from memory, the ram gets filled and the thing gets killed. Lots of ram seems to get filled up as soon as we've loaded the model, before we've even begun training.
* [ ] When trying to ctrl+c a model that's been training for a while and has almost filled up the ram, the model won't save right, and MuZero will crash and the model will lose all its training progress.
* [ ] When training with the same settings as the paper, the thing trains INCREDIBLY slowly. I don't want to compromise, but we may have to for it to be feasable to train this thing.
* [ ] When the replay_buffer is too large in any game config (e.g. 512 in atari), the program will crash my entire computer after a short period of time while training.

## TODO

* [ ] Figure out Leviathan's terminal.
* [ ] Figure out Leviathan's web browser.

# Duvaud's Original Readme

A commented and [documented](https://github.com/werner-duvaud/muzero-general/wiki/MuZero-Documentation) implementation of MuZero based on the Google DeepMind [paper](https://arxiv.org/abs/1911.08265) (Nov 2019) and the associated [pseudocode](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py).
It is designed to be easily adaptable for every games or reinforcement learning environments (like [gym](https://github.com/openai/gym)). You only need to add a [game file](https://github.com/werner-duvaud/muzero-general/tree/master/games) with the hyperparameters and the game class. Please refer to the [documentation](https://github.com/werner-duvaud/muzero-general/wiki/MuZero-Documentation) and the [example](https://github.com/werner-duvaud/muzero-general/blob/master/games/cartpole.py).

MuZero is a state of the art RL algorithm for board games (Chess, Go, ...) and Atari games.
It is the successor to [AlphaZero](https://arxiv.org/abs/1712.01815) but without any knowledge of the environment underlying dynamics. MuZero learns a model of the environment and uses an internal representation that contains only the useful information for predicting the reward, value, policy and transitions. MuZero is also close to [Value prediction networks](https://arxiv.org/abs/1707.03497). See [How it works](https://github.com/werner-duvaud/muzero-general/wiki/How-MuZero-works).

## Features

* [x] Residual Network and Fully connected network in [PyTorch](https://github.com/pytorch/pytorch)
* [x] Multi-Threaded/Asynchronous/[Cluster](https://docs.ray.io/en/latest/cluster-index.html) with [Ray](https://github.com/ray-project/ray)
* [X] Multi GPU support for the training and the selfplay
* [x] TensorBoard real-time monitoring
* [x] Model weights automatically saved at checkpoints
* [x] Single and two player mode
* [x] Commented and [documented](https://github.com/werner-duvaud/muzero-general/wiki/MuZero-Documentation)
* [x] Easily adaptable for new games
* [x] [Examples](https://github.com/werner-duvaud/muzero-general/blob/master/games/cartpole.py) of board games, Gym and Atari games (See [list of implemented games](https://github.com/werner-duvaud/muzero-general#games-already-implemented))
* [x] [Pretrained weights](https://github.com/werner-duvaud/muzero-general/tree/master/results) available
* [ ] Windows support (Experimental / Workaround: Use the [notebook](https://github.com/werner-duvaud/muzero-general/blob/master/notebook.ipynb) in [Google Colab](https://colab.research.google.com))

### Further improvements
These improvements are active research, they are personal ideas and go beyond MuZero paper. We are open to contributions and other ideas.

* [x] [Hyperparameter search](https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization)
* [x] [Continuous action space](https://github.com/werner-duvaud/muzero-general/tree/continuous)
* [x] [Tool to understand the learned model](https://github.com/werner-duvaud/muzero-general/blob/master/diagnose_model.py)
* [ ] Support of stochastic environments
* [ ] Support of more than two player games
* [ ] RL tricks (Never Give Up,  Adaptive Exploration, ...)

## Demo

All performances are tracked and displayed in real time in [TensorBoard](https://www.tensorflow.org/tensorboard) :

![cartpole training summary](https://github.com/werner-duvaud/muzero-general/blob/master/docs/cartpole-training-summary.png)

Testing Lunar Lander :

![lunarlander training preview](https://github.com/werner-duvaud/muzero-general/blob/master/docs/lunarlander-training-preview.png)

## Games already implemented

* Cartpole      (Tested with the fully connected network)
* Lunar Lander  (Tested in deterministic mode with the fully connected network)
* Gridworld     (Tested with the fully connected network)
* Tic-tac-toe   (Tested with the fully connected network and the residual network)
* Connect4      (Slightly tested with the residual network)
* Gomoku
* Twenty-One / Blackjack    (Tested with the residual network)
* Atari Breakout

Tests are done on Ubuntu with 16 GB RAM / Intel i7 / GTX 1050Ti Max-Q. We make sure to obtain a progression and a level which ensures that it has learned. But we do not systematically reach a human level. For certain environments, we notice a regression after a certain time. The proposed configurations are certainly not optimal and we do not focus for now on the optimization of hyperparameters. Any help is welcome.

## Code structure

![code structure](https://github.com/werner-duvaud/muzero-general/blob/master/docs/code-structure-werner-duvaud.png)

Network summary:

<p align="center">
<a href="https://github.com/werner-duvaud/muzero-general/blob/master/docs/muzero-network-werner-duvaud.png">
<img src="https://github.com/werner-duvaud/muzero-general/blob/master/docs/muzero-network-werner-duvaud.png" width="250"/>
</a>
</p>

## Getting started
### Installation

```bash
git clone https://github.com/werner-duvaud/muzero-general.git
cd muzero-general

pip install -r requirements.txt
```

### Run

```bash
python muzero.py
```
To visualize the training results, run in a new terminal:
```bash
tensorboard --logdir ./results
```

### Config

You can adapt the configurations of each game by editing the `MuZeroConfig` class of the respective file in the [games folder](https://github.com/werner-duvaud/muzero-general/tree/master/games).

## Authors

* Werner Duvaud
* Aurèle Hainaut
* Paul Lenoir
* [Contributors](https://github.com/werner-duvaud/muzero-general/graphs/contributors)

Please use this bibtex if you want to cite this repository (master branch) in your publications:
```bash
@misc{muzero-general,
  author       = {Werner Duvaud, Aurèle Hainaut},
  title        = {MuZero General: Open Reimplementation of MuZero},
  year         = {2019},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/werner-duvaud/muzero-general}},
}
```

## Getting involved

* [GitHub Issues](https://github.com/werner-duvaud/muzero-general/issues): For reporting bugs.
* [Pull Requests](https://github.com/werner-duvaud/muzero-general/pulls): For submitting code contributions.
* [Discord server](https://discord.gg/GB2vwsF): For discussions about development or any general questions.
