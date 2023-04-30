import torch 
import torch.nn as nn

import numpy as np


class Actor(nn.Module):

    def __init__(self, dim_states, dim_actions, continuous_control):
        super(Actor, self).__init__()
        # MLP, fully connected layers, ReLU activations, linear ouput activation
        # dim_states -> 64 -> 64 -> dim_actions

        if continuous_control:
            # trainable parameter
            self._log_std = None


    def forward(self, input):
        return input


class Critic(nn.Module):

    def __init__(self, dim_states):
        super(Critic, self).__init__()
        # MLP, fully connected layers, ReLU activations, linear ouput activation
        # dim_states -> 64 -> 64 -> 1


    def forward(self, input):
        return input


class ActorCriticAgent:

    def __init__(self, dim_states, dim_actions, actor_lr, critic_lr, gamma, continuous_control=False):
        
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        self._gamma = gamma

        self._dim_states = dim_states
        self._dim_actions = dim_actions

        self._continuous_control = continuous_control

        self._actor = Actor(self._dim_states, self._dim_actions, self._continuous_control)

        # Adam optimizer
        self._actor_optimizer = None

        self._critic = Critic(self._dim_states)

        # Adam optimizer
        self._critic_optimizer = None

        self._select_action = self._select_action_continuous if self._continuous_control else self._select_action_discrete
        self._compute_actor_loss = self._compute_actor_loss_continuous if self._continuous_control else self._compute_actor_loss_discrete


    def select_action(self, observation):
        return self._select_action(observation)
        

    def _select_action_discrete(self, observation):
        # sample from categorical distribution
        pass

    def _select_action_continuous(self, observation):
        # sample from normal distribution
        # use the log std trainable parameter
        pass


    def _compute_actor_loss_discrete(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        pass


    def _compute_actor_loss_continuous(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        pass


    def _compute_critic_loss(self, observation_batch, reward_batch, next_observation_batch, done_batch):
        # minimize mean((r + gamma * V(s_t1) - V(s_t))^2)
        pass


    def update_actor(self, observation_batch, action_batch, reward_batch, next_observation_batch, done_batch):
        # compute the advantages using the critic and update the actor parameters
        # use self._compute_actor_loss
        pass
        
        
    def update_critic(self, observation_batch, reward_batch, next_observation_batch, done_batch):
        # update the critic
        # use self._compute_critic_loss
        pass