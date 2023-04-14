import torch
import torch.nn as nn

import copy

import numpy as np

from replay_buffer import ReplayBuffer


class DeepQNetwork(nn.Module):

    def __init__(self, dim_states, dim_actions):
        super(DeepQNetwork, self).__init__()
        # MLP, fully connected layers, ReLU activations, linear ouput activation
        # dim_states -> 64 -> 64 -> dim_actions

    def forward(self, input):
        return input
    

class DeepQNetworkAgent:

    def __init__(self, dim_states, dim_actions, lr, gamma, epsilon, nb_training_steps, replay_buffer_size, batch_size):
        
        self._learning_rate = lr
        self._gamma = gamma
        self._epsilon = epsilon

        self._epsilon_min = 0
        self._epsilon_decay = self._epsilon / (nb_training_steps / 2.)

        self._dim_states = dim_states
        self._dim_actions = dim_actions

        self.replay_buffer = ReplayBuffer(dim_states=self._dim_states,
                                          dim_actions=self._dim_actions,
                                          max_size=replay_buffer_size,
                                          sample_size=batch_size)

        # Complete
        self._deep_qnetwork = DeepQNetwork(self._dim_states, self._dim_actions)
        self._target_deepq_network = None

        # Adam optimizer
        self._optimizer = None


    def store_transition(self, s_t, a_t, r_t, s_t1, done_t):
        self.replay_buffer.store_transition(s_t, a_t, r_t, s_t1, done_t)


    def replace_target_network(self):
        pass


    def select_action(self, observation, greedy=False):

        if np.random.random() > self._epsilon or greedy:
            # Select action greedily
            pass

        else:
            # Select random action
            pass

        if not greedy and self._epsilon >= self._epsilon_min:
            # Implement epsilon linear decay
            pass
        
        return action


    def update(self):
        pass
