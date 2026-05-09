"""
This file contains SAGEPolicy which implements actor and critic networks
with SAGE-style topological guidance (ASC validity masking + TNSC potentials).
"""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def initialize_layer(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initializes the weights and biases of a given layer.

    Parameters:
    layer (nn.Module): The neural network layer to initialize.
    std (float): The standard deviation for orthogonal initialization of weights. Default is sqrt(2).
    bias_const (float): The constant value to initialize the biases. Default is 0.0.

    Returns:
    nn.Module: The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def build_network(nodes_counts, std=0.01):
    """
    Constructs a neural network with fully connected layers and Tanh activations based on the specified node counts.

    Parameters:
    nodes_counts (list of int): A list where each element represents the number of nodes in a layer.
    std (float): The standard deviation for initializing the final layer's weights. Default is 0.01.

    Returns:
    list: A list of layers (including activation functions) representing the neural network.
    """
    layers = [initialize_layer(nn.Linear(nodes_counts[0], nodes_counts[1])), nn.Tanh()]

    for i in range(1, len(nodes_counts) - 2):
        layers.append(initialize_layer(nn.Linear(nodes_counts[i], nodes_counts[i + 1])))
        layers.append(nn.Tanh())

    layers.append(initialize_layer(nn.Linear(nodes_counts[-2], nodes_counts[-1]), std=std))

    return layers


class SAGEPolicy(nn.Module):
    """
    Actor-critic policy/value model for SAGE, with optional guided sampling inputs.
    """

    def __init__(self, envs, nodes_counts):
        super().__init__()

        input_dim = np.prod(envs.single_observation_space.shape)
        self.critic_nodes = [input_dim] + nodes_counts + [1]
        self.actor_nodes = [input_dim] + nodes_counts + [envs.single_action_space.n]

        self.critic = nn.Sequential(*build_network(self.critic_nodes, 1.0))
        self.actor = nn.Sequential(*build_network(self.actor_nodes, 0.01))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(
        self,
        x,
        action=None,
        valid_action_mask=None,
        psi_total=None,
        trex_lambda=1.0,
    ):
        logits = self.actor(x)
        value = self.critic(x)

        # Apply SAGE guidance if provided
        if valid_action_mask is not None and psi_total is not None:
            # Convert to tensors if needed and ensure correct device/dtype
            if not isinstance(valid_action_mask, torch.Tensor):
                valid_mask_tensor = torch.tensor(
                    valid_action_mask, dtype=torch.bool, device=x.device
                )
            else:
                valid_mask_tensor = valid_action_mask.to(x.device)

            if not isinstance(psi_total, torch.Tensor):
                psi_tensor = torch.tensor(psi_total, dtype=torch.float32, device=x.device)
            else:
                psi_tensor = psi_total.to(x.device)

            logits = logits.clone()
            logits[~valid_mask_tensor] = float("-inf")
            logits = logits + trex_lambda * psi_tensor

        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value

