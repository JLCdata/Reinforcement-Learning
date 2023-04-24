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

        # Inicialización de pesos con distribución uniforme
        def init_weights(m):
            if isinstance(m, nn.Linear):

                #nn.init.normal_(m.weight, mean=0.0, std=1)
                #nn.init.normal_(m.weight, mean=0.0, std=1)
                nn.init.uniform_(m.weight, a=0.0, b=1.0)
                nn.init.uniform_(m.bias, a=0.0, b=1.0)
                
        self.layers = nn.Sequential(
            nn.Linear(dim_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dim_actions)
        )

        self.layers.apply(init_weights)

    def forward(self, input):

        # tensor format
        if isinstance(input, torch.Tensor):
            input=input
            
        else:
            input = torch.from_numpy(input).unsqueeze(dim=0).float()
            
        q_values = self.layers(input)

        return q_values

class DeepQNetworkAgent:

    def __init__(self, dim_states, dim_actions, lr, gamma, epsilon, nb_training_steps, replay_buffer_size, batch_size):
        
        self._learning_rate = lr
        self._gamma = gamma
        self._epsilon = epsilon

        self._epsilon_min = 0.0
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
                qa = self._deep_qnetwork(observation)

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
        s_t,a_t,r_t,s_t1,done=self.replay_buffer.sample_transitions()

        s_t=torch.from_numpy(s_t).unsqueeze(dim=0).float()
        s_t1=torch.from_numpy(s_t1).unsqueeze(dim=0).float()
        r_t = torch.tensor(r_t).view(-1, 1).float()
        done = torch.tensor(done).view(-1, 1)
        a_t=torch.tensor(a_t).view(-1, 1).type(torch.int64)

        # Predict Q-value de estado actual
        qsa_predict=self._deep_qnetwork(s_t)
        qsa_actions=torch.gather(input=qsa_predict[0], dim=1,index = a_t)
        
        # Calculo de Q-value target (Q-value estado siguiente)
        next_qsa_predict=self._target_deepq_network(s_t1)
        max_next_qsa_predict=torch.max(next_qsa_predict, dim=-1, keepdim=True)[0][0]

        target_qsa=r_t+~done*self._gamma*max_next_qsa_predict

        loss = F.mse_loss(qsa_actions, target_qsa)
        self._deep_qnetwork.zero_grad()
        loss.backward()
        self._optimizer.step()

