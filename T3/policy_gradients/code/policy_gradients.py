import torch 
import torch.nn as nn

import numpy as np
from torch.optim import AdamW

class Policy(nn.Module):
    
    def __init__(self, dim_states, dim_actions, continuous_control):
        super(Policy, self).__init__()
        # MLP, fully connected layers, ReLU activations, linear ouput activation
        # dim_states -> 64 -> 64 -> dim_actions

        self.layers = nn.Sequential(
            nn.Linear(dim_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dim_actions)
        )
        
        if continuous_control:
            # trainable parameter
            self.log_std = nn.Parameter(torch.zeros(1, dim_actions))


    def forward(self, input):

        # tensor format
        if isinstance(input, torch.Tensor):
            input=input
            
        else:
            input = torch.from_numpy(input).unsqueeze(dim=0).float()
            
        value = self.layers(input)
        
        return value


class PolicyGradients:

    def __init__(self, dim_states, dim_actions, lr, gamma, 
                 continuous_control=False, reward_to_go=False, use_baseline=False):
        
        self._learning_rate = lr
        self._gamma = gamma
        
        self._dim_states = dim_states
        self._dim_actions = dim_actions

        self._continuous_control = continuous_control
        self._use_reward_to_go = reward_to_go
        self._use_baseline = use_baseline

        self._policy = Policy(self._dim_states, self._dim_actions, self._continuous_control)
        # Adam optimizer
        self._optimizer = AdamW(self._policy.parameters(), lr=self._learning_rate)

        self._select_action = self._select_action_continuous if self._continuous_control else self._select_action_discrete
        self._compute_loss = self._compute_loss_continuous if self._continuous_control else self._compute_loss_discrete


    def select_action(self, observation):
        return self._select_action(observation)
        

    def _select_action_discrete(self, observation):
        # sample from categorical distribution
        RN_policy=self._policy 
        logits=RN_policy(observation)

        # Probabilidad de cada acción
        probs = torch.softmax(logits, dim=-1)

        # Distribución de probabilidad categorica
        dist = torch.distributions.Categorical(probs)

        # Sample de acción
        action = dist.sample()#.item()
     
        return action


    def _select_action_continuous(self, observation):
        # sample from normal distribution
        # use the log std trainable parameter

        # RN
        RN_policy=self._policy

        # Parametro log std de la RN
        log_std=RN_policy.log_std
        std = torch.exp(log_std)

        # Politica dada la observación (Representa el promedio de la distribución normal que muestrea acciones)
        policy=RN_policy(observation)
        
        # Distribución normal de parametros mean y std, esta se utiliza para muestrear acciones de modo de tal de explorar el espacio de acciones
        dist = torch.distributions.Normal(policy, std)

        # sample de acción
        action = dist.sample()
        
        # Asegurarse de que las acciones están dentro del rango [-1, 1]
        #action = torch.tanh(action)
        
        return action
            

    def update(self, observation_batch, action_batch, advantage_batch):
        # update the policy here

        # you should use self._compute_loss 
        loss=-self._compute_loss(observation_batch, action_batch, advantage_batch)
        
        # Backpropagation
        self._policy.zero_grad()
        loss.backward()
        self._optimizer.step()
    

    def _compute_loss_discrete(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        logits=self._policy(observation_batch)

        # Distribución de probabilidad categorica
        dist = torch.distributions.Categorical(logits=logits)
        log_probs=dist.log_prob(torch.tensor(action_batch)).squeeze(0)
        retorno_descontado=torch.tensor(advantage_batch)
        loss=torch.mean(log_probs*retorno_descontado)#.item()
        return loss
        


    def _compute_loss_continuous(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        means=self._policy(observation_batch).squeeze(0).squeeze(1)

        log_std=self._policy.log_std
        std = torch.exp(log_std)

        # Distribución normal de parametros mean y std, esta se utiliza para muestrear acciones de modo de tal de explorar el espacio de acciones
        dist = torch.distributions.Normal(means, std)
        log_probs=dist.log_prob(torch.tensor((action_batch))).squeeze(0)
        retorno_descontado=torch.tensor(advantage_batch)
        loss=torch.mean(log_probs*retorno_descontado)#.item()
        return loss
        
    def estimate_returns(self, rollouts_rew):
        estimated_returns = []
        for rollout_rew in rollouts_rew:

            # Largo del episodio (largo del reward)
            n_steps = len(rollout_rew)
            estimated_return = np.zeros(n_steps)

            if self._use_reward_to_go:
                # only for part 2
                vec_gammas=np.array([self._gamma**j for j in range(n_steps)])

                for t in range(n_steps):

                    sum_descount=np.sum(vec_gammas[t:]*rollout_rew[t:])
                    estimated_return[t] = sum_descount
            else:

                estimated_return = np.zeros(n_steps)

                vec_gammas=np.array([self._gamma**j for j in range(n_steps)])

                sum_descount=np.sum(vec_gammas*rollout_rew)

                for t in range(n_steps):
                    
                    estimated_return[t] = sum_descount
                 
            
            estimated_returns = np.concatenate([estimated_returns, estimated_return])

        if self._use_baseline:
            # only for part 2
            average_return_baseline = np.mean(estimated_returns)
            estimated_returns -= average_return_baseline

        return np.array(estimated_returns, dtype=np.float32)

