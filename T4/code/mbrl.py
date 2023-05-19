import torch
import torch.nn as nn

import numpy as np

from buffer import Buffer

class Model(nn.Module):

    def __init__(self, dim_states, dim_actions, continuous_control):
        super(Model, self).__init__()
        """if continuous_control:
            self._fc1 = None
        else:
            self._fc1 = None
        """

        # MLP, fully connected layers, ReLU activations, linear ouput activation
        # dim_input -> 64 -> 64 -> dim_actions


    def forward(self, state, action):
        return None


class RSPlanner:

    def __init__(self, dim_states, dim_actions, continuous_control, model, planning_horizon, nb_trajectories, reward_function):
        self._dim_states = dim_states
        self._dim_actions = dim_actions
        self._continuous_control = continuous_control

        self._model = model

        self._planning_horizon = planning_horizon
        self._nb_trajectories = nb_trajectories
        self._reward_function = reward_function

        
    def generate_plan(self, observation):
        # Generate a sequence of random actions
        if self._continuous_control:
            random_actions = None
        else:
            random_actions = None
        
        # Construct initial observation 
        o_t = None

        rewards = torch.zeros((self._nb_trajectories, ))
        for i in range(self._planning_horizon):
            # Get a_t
            if self._continuous_control:
                a_t = None
            else:
                a_t = None

            # Predict next observation using the model

            # Compute reward (use reward_function)
            
            o_t = o_t1

        # Return the best sequence of actions
        return None




class MBRLAgent:

    def __init__(self, dim_states, dim_actions, continuous_control, model_lr, buffer_size, batch_size, 
                       planning_horizon, nb_trajectories, reward_function):

        self._dim_states = dim_states
        self._dim_actions = dim_actions

        self._continuous_control = continuous_control

        self._model_lr = model_lr

        self._model = Model(self._dim_states, self._dim_actions, self._continuous_control)

        # Adam optimizer
        self._model_optimizer = None

        self._buffer = Buffer(self._dim_states, self._dim_actions, buffer_size, batch_size)
        
        self._planner = RSPlanner(self._dim_states, self._dim_actions, self._continuous_control, 
                                  self._model, planning_horizon, nb_trajectories, reward_function)


    def select_action(self, observation, random=False):

        if random:
            # Return random action
            if self._continuous_control:
                return None
            return None

        # Generate plan
        plan = None

        # Return the first action of the plan
        if self._continuous_control:
            return None
        
        return None


    def store_transition(self, s_t, a_t, s_t1):
        pass


    def update_model(self):
        batches = self._buffer.get_batches()
        for batch in batches:
            # Use the batches to train the model
            # loss: avg((s_t1 - model(s_t, a_t))^2)
            pass
        