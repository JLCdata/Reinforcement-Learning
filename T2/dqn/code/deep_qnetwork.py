import torch
import torch.nn as nn

import copy

import numpy as np

from replay_buffer import ReplayBuffer

from torch.optim import AdamW

import torch.nn.functional as F

class DeepQNetwork(nn.Module):

    def __init__(self, dim_states, dim_actions):
        super(DeepQNetwork, self).__init__()
        # MLP, fully connected layers, ReLU activations, linear ouput activation
        # dim_states -> 64 -> 64 -> dim_actions
        self.layers = nn.Sequential(
            nn.Linear(dim_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dim_actions)
        )

    def forward(self, input):
        # tensor format
        #input = torch.from_numpy(input).unsqueeze(dim=0).float()

        q_values = self.layers(input)

        return q_values

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
        self._target_deepq_network = copy.deepcopy(self._deep_qnetwork).eval()

        # Adam optimizer
        self._optimizer = AdamW(self._deep_qnetwork.parameters(), lr=self._learning_rate)


    def store_transition(self, s_t, a_t, r_t, s_t1, done_t):
        self.replay_buffer.store_transition(s_t, a_t, r_t, s_t1, done_t)


    def replace_target_network(self):
        self._target_deepq_network.load_state_dict(self._deep_qnetwork.state_dict())
        

    def select_action(self, observation, greedy=False):
        
           
            if np.random.random() > self._epsilon or greedy:
                # Select action greedily

                # Action values
                qa = self._target_deepq_network(observation)

                # Action con mayor q-value
                action=qa.argmax().item()
        
            else:
                # Exploración
                action=np.random.randint(2)

            if not greedy and self._epsilon >= self._epsilon_min:

                # Implement epsilon linear decay
                self._epsilon-=self._epsilon_decay 
                

            return action

    def update(self):
        s_t, a_t, r_t, s_t1, done_t=self.replay_buffer.sample_transitions()

        qsa_b = q_network(state_b).gather(1, action_b)
                
        next_qsa_b = target_q_network(next_state_b)
        next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]
        
        target_b = reward_b + ~done_b * gamma * next_qsa_b
        loss = F.mse_loss(qsa_b, target_b)
        q_network.zero_grad()
        loss.backward()
        optim.step()
        pass
