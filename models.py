from abc import ABC, abstractmethod

import torch


# This class is used to create all of MuZero's neural networks - the representation function, prediction function, and dynamics function.
# With the game config file, you can choose whether you want those networks to be fully connected or residual nns.
class MuZeroNetwork:
    # Here's an explanation of what __new__ is in python: https://howto.lintel.in/python-__new__-magic-method-explained/
    def __new__(cls, config):
        # In a game's config file, you can set the network to either be fullyconnected or resnet. So far I don't really understand the difference between the two. MuZero, to my knowledge, uses CNNs.
        # Perhaps, for my purposes, I could use a transformer? Would that even make any sense? Guess we'll find out...
        if config.network == "fullyconnected":
            return MuZeroFullyConnectedNetwork(  # MuZeroFullyConnectedNetwork() defined below, where I further elaborate on how the arguments are used. Takes a TON of arguments from the game's config file.
                config.observation_shape,        # So really, all the meat of the way a network is defined is in the game file.
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,
                config.fc_reward_layers,         # So are there actually FIVE neural networks?? I thought there were three...
                config.fc_value_layers,
                config.fc_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.support_size,
            )
        elif config.network == "resnet":
            return MuZeroResidualNetwork(        # Same as above, but with resnet.
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.blocks,
                config.channels,
                config.reduced_channels_reward,
                config.reduced_channels_value,
                config.reduced_channels_policy,
                config.resnet_fc_reward_layers,
                config.resnet_fc_value_layers,
                config.resnet_fc_policy_layers,
                config.support_size,
                config.downsample,
            )
        else:
            raise NotImplementedError('The network parameter you entered has not yet been implemented. It should be "fullyconnected" or "resnet".')


# Transfer all tensor objects in a dictionary... to another dictionary?
def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):  # If the value is a tensor, add it to CPU memory.
            cpu_dict[key] = value.cpu()  # .cpu() Returns a copy of this tensor object in CPU memory.
        elif isinstance(value, dict):  # If the value is a dictionary, recursively call the function.
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


# ABC = Abstract Base Class. torch.nn.Module is the base class for all neural network modules. Your models should also subclass this class.
class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod  # Put this @ decorator in front of abstract methods. Don't ask why just do it.
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())  # .state_dict() is from torch.nn.Module. Returns a dictionary containing a whole state of the module.

    def set_weights(self, weights):  # weights should be a dictionary.
        self.load_state_dict(weights)


#####################################################
######## Fully Connected Network Definition #########


class MuZeroFullyConnectedNetwork(AbstractNetwork):
    def __init__(  # Takes all the same arguments we put in the MuZeroNetwork class above.
        self,
        observation_shape,  # Something to do with the shape of the input of the representation function.
        stacked_observations,  # Number of previous observations and previous actions to add to the current observation
        action_space_size,  # The size of the action space. Will probably be about 94 for me? Or maybe 188. It depends on how the environment will work.
        encoding_size,  # The size of the output of the representation function's hidden state network, and the size of the input of the dynamics and prediction functions' neural nets.
        fc_reward_layers,  # How many layers are in the reward function.
        fc_value_layers,   # How many layers are in the value function.
        fc_policy_layers,  # How many layers are in the policy function.
        fc_representation_layers,  # How many layers are in the representation function.
        fc_dynamics_layers,  # How many layers are in the dynamics function.
        support_size,  # Something from the MuZero paper I think, just treat it as magic incantation for now I guess.
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1  # I'm not sure where the 2 * X + 1 comes from. Magic incantation.

        # Pretty self explanatory, this is the representation network used by the representation function to encode/map the environment's state into a hidden state. I think.
        self.representation_network = torch.nn.DataParallel(  # Wrapping the multilayer perceptron in .DataParallel() allows us to parallelize it. Official docs say "It is recommended to use DistributedDataParallel, instead of this class, to do multi-GPU training"
            mlp(  # Multilayer perceptron, defined at the very bottom for some reason. I went down to the bottom and coded this next, so you might want to go down too and see how it works. Despite all the lines below, we're only using 3 arguments here, the first being the neural net's input size.
                observation_shape[0]  # I don't like to spread out the arguments of a function across multiple lines like this because I think it looks confusing, but I guess here it's necessary because of the large expression being used as an argument.
                * observation_shape[1]
                * observation_shape[2]
                * (stacked_observations + 1)
                + stacked_observations * observation_shape[1] * observation_shape[2],  # I'm not exactly sure what all this adds up to. Magic incantation again, I guess.
                fc_representation_layers,
                encoding_size,
            )
        )

        # Takes the encoded/hidden state from the representation function so it can be put in the dynamics function.
        # Wait, but this is isn't using the representation network, this is a completely new neural net. So does the dynamics function have TWO neural networks? one for the reward and one for the next state?
        # That could really complicate things, because if all of the agent's knowledge from the internet is in the representation function, how is this NN going to be useful if it doesn't have any of that knowledge? Does it?
        # Oh wait! So INITIALLY, it takes the hidden state from the representation function as input, but then afterward it takes the hidden state from previous uses of the dynamics function.
        # So since it takes the hidden state from the representation function initially, it WILL have the information from the internet that it needs to write code. Ok, that should work great, then!
        self.dynamics_encoded_state_network = torch.nn.DataParallel(
            mlp(encoding_size + self.action_space_size, fc_dynamics_layers, encoding_size)  # FIXME I don't think the comma in Duvaud's code is necessary here.
        )

        # The dynamics function's reward neural net.
        self.dynamics_reward_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_reward_layers, self.full_support_size)
        )

        # The policy neural net.
        self.prediction_policy_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_policy_layers, self.action_space_size)  # The policy net uses the action space size to determine the size of the output. Makes sense, right?
        )

        # The value neural net.
        self.prediction_value_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_value_layers, self.full_support_size)
        )
        # Both the policy nn and the value nn are used in the prediction function.

    # The prediction function. I didn't know it was literally a function. Returns the policy network and the value network.
    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)  # What's a "logit"? Google says "The logit link function is used to model the probability of 'success' as a function of covariates (e.g., logistic regression)."
        # So I think in this case, policy_logits is the prediction function's policy neural network. I mean, that's what the code looks like.
        value = self.prediction_value_network(encoded_state)  # Similar to above, seems to be the prediction function's value neural network.
        return policy_logits, value
    # Not sure if it matters which order you define the function in in the code, but just to be clear, in MuZero, I'm pretty sure the representation function is used BEFORE the prediction function.

    # The representation function, returns the hidden/encoded state that the prediction and dynamics functions can understand and use.
    def representation(self, observation):
        # most of this is voodoo magic, talked about in Appendix G (Training) of the MuZero paper.
        encoded_state = self.representation_network(observation.view(observation.shape[0], -1))  # So I had to do a ton of ctrl-clicking to figure out what the observations is supposed to be, but it seems to be defined by the game file of whatever game muzero is playing. Magic incantation for now.
        # Scale encoded state between [0, 1] (See the training appendix of the MuZero paper)
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]  # Basically, returns the minimum value of all elements in the input tensor. A little different though because of the arguments, take a look: https://pytorch.org/docs/stable/generated/torch.min.html
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]  # Same as above, but returns max: https://pytorch.org/docs/stable/generated/torch.max.html
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (encoded_state - min_encoded_state) / scale_encoded_state
        return encoded_state_normalized

    # The dynamics function, returns the reward as well as what MuZero thinks the next state will be based on the action taken.
    # We may need to rewrite
    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))  # torch.zeros returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size. I don't know what action.shape does.
            .to(action.device)  # torch.Tensor.to performs Tensor dtype and/or device conversion. So I think it converts the tensor to either your GPU or CPU depending on what you're using?
            .float()  # Turns the tensor to type float I think.
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)  # torch.cat concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.

        next_encoded_state = self.dynamics_encoded_state_network(x)  # After all that voodoo, take the encoded/hidden state from the representation function and put it in the dynamics function.

        reward = self.dynamics_reward_network(next_encoded_state)  # Use the hidden state from the representation function to create the reward nn.

        # ADE DUE DAMBALLA. GIVE ME THE POWER I BEG OF YOU.
        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state

        return next_encoded_state_normalized, reward

    # Used in self_play.py at the root of the MCTS search tree to figure out what to do with MuZero's first (aka initial) observation.
    def initial_inference(self, observation):
        encoded_state = self.representation(observation)  # Pretty self explanatory, if you've read the rest of the code.
        policy_logits, value = self.prediction(encoded_state)  # Putting the encoded state from the representation function in the prediction function and getting the policy & value from it.
        # reward equal to 0 for consistency. I don't know what that means, but this kind of looks like what we did above in the dynamics function I guess.
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        return value, reward, policy_logits, encoded_state

    # Used in self_play.py by the MCTS to obtain each hidden state after the initial observation.
    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state  # Why did he put the outputs of the last function in multiple lines in parentheses but not this one I'm so confused.

### So after implementing the fully connected network, I actually got MuZero to work! Kind of! It actually took me to the menu this time, although I couldn't train it or
### test it or anything like that because I hadn't implemented those functions in muzero.py, but hey, it was a step forward, at least! I could do some things like manually
### test the game and do stuff that didn't depend on stuff I hadn't implemented yet, so that's something. Changing the network type to "resnet" in the game file, of course,
### caused it to give me errors again, but I decided to implement that later and, for now, try to implement the .train() function back in muzero.py, just sticking with
### the fully connected network type for now. So if you're reading this in the chronological order that I've implemented it in, head over there now!

###### End Fully Connected #######
##################################


##################################
############# ResNet #############


# def conv3x3


########### End ResNet ###########
##################################


# Function that returns a multilayer perceptron, which is a sequential neural network.
def mlp(input_size, layer_sizes, output_size, output_activation=torch.nn.Identity, activation=torch.nn.ELU):  # Duvaud, again, spread these arguments over multiple lines in his code. To me, though, it's more intuitive and easy to read if they're all in one line. Plus its shorter, too.
    sizes = [input_size] + layer_sizes + [output_size]  # I'm not sure why the input_size and output_size are put into lists with only one item. Why cant they just be scalars? I guess it's so they can be used in the for loop?
    layers = []  # A list of layers to be put into the mlp.
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation  # So I think, for each neural layer in the network, this tells it to use 'activation' as the layer's activation function if it's NOT an output layer, and use output_activation if it is.
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]  # Add a linear layer to the list, with act() as the layer's activation function I think.
    return torch.nn.Sequential(*layers)  # Return a sequential neural network using the list of layers that we just created.


# Voodoo used by the MCTS to process the value and reward so that it can be stored by nodes and used in backpropagation.
def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar. See paper appendix Network Architecture.
    """
    # Decode to a scalar? I have no idea what any of this is for.
    probabilities = torch.softmax(logits, dim=1)
    support = torch.tensor([x for x in range(-support_size, support_size + 1)]).expand(probabilities.shape).float().to(device=probabilities.device)
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001)) ** 2 - 1)
    return x
### Alright, back to self_play.py!


# Voodoo used in trainer.py
def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1))
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits