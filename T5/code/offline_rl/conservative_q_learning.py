import torch
import torch.nn as nn

import copy

import numpy as np


class DeepQNetwork(nn.Module):

    def __init__(self, dim_states, dim_actions):
        super(DeepQNetwork, self).__init__()
        self._fc1 = nn.Linear(dim_states, 64)
        self._fc2 = nn.Linear(64, 64)
        self._fc3 = nn.Linear(64, dim_actions)
        self._relu = nn.functional.relu

    def forward(self, input):
        output = self._relu(self._fc1(input))
        output = self._relu(self._fc2(output))
        output = self._fc3(output)
        return output
    

class ConservativeDeepQNetworkAgent:

    def __init__(self, dim_states, dim_actions, lr, gamma, alpha):
        
        self._learning_rate = lr
        self._gamma = gamma
        self._alpha = alpha

        self._dim_states = dim_states
        self._dim_actions = dim_actions

        self._deep_qnetwork = DeepQNetwork(self._dim_states, self._dim_actions)
        self._target_deepq_network = copy.deepcopy(self._deep_qnetwork)

        self._optimizer = torch.optim.Adam(self._deep_qnetwork.parameters(), lr=self._learning_rate)


    def replace_target_network(self):
        with torch.no_grad():
            for param, target_param in zip(self._deep_qnetwork.parameters(), 
                                           self._target_deepq_network.parameters()):
                target_param.data = param


    def select_action(self, observation, greedy=False):

        # Select action greedily
        with torch.no_grad():
            action = self._deep_qnetwork(torch.from_numpy(np.array(observation, dtype=np.float32))).argmax().numpy()

        return action

    '''
    def update(self, experiences_batch):

        s_t_batch, a_t_batch, r_t_batch, s_t1_batch, done_t_batch = experiences_batch

        # P2 - 1
        
        with torch.no_grad():
            target = None


        self._optimizer.zero_grad()
        # CQL1 loss (check the paper)
        # hint: use torch.logsumexp
        c_target = None

        loss = None # DQN loss + self._alpha * c_target

        loss.backward()

        self._optimizer.step()
    '''
    
    def update(self, experiences_batch):
    
        s_t_batch, a_t_batch, r_t_batch, s_t1_batch, done_t_batch = experiences_batch

        # Q-values for current state-action pairs
        q_values = self._deep_qnetwork(torch.from_numpy(s_t_batch)).gather(1, torch.tensor(a_t_batch, dtype=torch.int64).reshape(-1,1))

        with torch.no_grad():
            # Q-values for next state-action pairs (from the target network)
            next_q_values = self._target_deepq_network(torch.from_numpy(s_t1_batch))
            max_next_q_values = torch.max(next_q_values, dim=1, keepdim=True)[0]

        target_q_values = torch.from_numpy(r_t_batch) + (1 - torch.from_numpy(done_t_batch)) * self._gamma * max_next_q_values

        # DQN loss
        dqn_loss = nn.MSELoss()(q_values, target_q_values)

        c_target = torch.logsumexp(self._deep_qnetwork(torch.from_numpy(s_t_batch)), dim=1).mean()-q_values.mean()
        c_target = c_target.mean()

        loss = dqn_loss + self._alpha * c_target

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    '''
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

        loss = F.mse_loss(qsa_actions, target_qsa)+alpha*C
        self._deep_qnetwork.zero_grad()
        loss.backward()
            self._optimizer.step()
    '''