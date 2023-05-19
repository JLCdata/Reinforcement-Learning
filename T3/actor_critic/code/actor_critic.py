import torch 
import torch.nn as nn

import numpy as np
from torch.optim import AdamW
import torch.nn.functional as F

# Actor
class Actor(nn.Module):
    
    def __init__(self, dim_states, dim_actions, continuous_control):
        super(Actor, self).__init__()
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


class Critic(nn.Module):
    
    def __init__(self, dim_states):
        super(Critic, self).__init__()
        # MLP, fully connected layers, ReLU activations, linear ouput activation
        # dim_states -> 64 -> 64 -> 1

       

        self.layers = nn.Sequential(
            nn.Linear(dim_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        
        
    def forward(self, input):
    
        # tensor format
        if isinstance(input, torch.Tensor):
            input=input
            
        else:
            input = torch.from_numpy(input).unsqueeze(dim=0).float()
            
        value = self.layers(input)
        
        return value


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
        self._actor_optimizer = AdamW(self._actor.parameters(), lr=self._actor_lr)

        self._critic = Critic(self._dim_states)

        # Adam optimizer
        self._critic_optimizer = AdamW(self._critic.parameters(), lr=self._critic_lr)

        self._select_action = self.select_action_continuous if self._continuous_control else self.select_action_discrete
        self._compute_actor_loss = self.compute_actor_loss_continuous if self._continuous_control else self.compute_actor_loss_discrete


    def select_action(self, observation):
        return self._select_action(observation)
        

    def select_action_discrete(self, observation):
        # sample from categorical distribution
         
        logits=self._actor(observation)#.detach()

        # Probabilidad de cada acción
        probs = torch.softmax(logits, dim=-1)

        # Distribución de probabilidad categorica
        dist = torch.distributions.Categorical(probs)

        # Sample de acción
        action = dist.sample()#.item()
     
        return action


    def select_action_continuous(self, observation):
        # sample from normal distribution
        # use the log std trainable parameter

        # Parametro log std de la RN
        log_std=self._actor.log_std#.detach()
        std = torch.exp(log_std)

        # Politica dada la observación (Representa el promedio de la distribución normal que muestrea acciones)
        means=self._actor(observation)#.detach()
        
        # Distribución normal de parametros mean y std, esta se utiliza para muestrear acciones de modo de tal de explorar el espacio de acciones
        dist = torch.distributions.Normal(means, std)

        # sample de acción
        action = dist.sample()
        
        return action


    def compute_actor_loss_discrete(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        logits=self._actor(observation_batch)
        #print(logits)
        # Distribución de probabilidad categorica
        dist = torch.distributions.Categorical(logits=logits)
        log_probs=dist.log_prob(torch.tensor(action_batch)).squeeze(0)
        #print(log_probs)
        advantage=torch.tensor(advantage_batch)
        #print(advantage)
        loss=torch.mean(log_probs*advantage)#.item()
        #print(loss)
        return loss


    def compute_actor_loss_continuous(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        means=self._actor(observation_batch).squeeze(0).squeeze(1)

        log_std=self._actor.log_std
        std = torch.exp(log_std)

        # Distribución normal de parametros mean y std, esta se utiliza para muestrear acciones de modo de tal de explorar el espacio de acciones
        dist = torch.distributions.Normal(means, std)
        log_probs=dist.log_prob(torch.tensor((action_batch))).squeeze(0)
        advantage=torch.tensor(advantage_batch)
        #print(log_probs)
        #print(advantage)
        loss=torch.mean(log_probs*advantage)#.item()
        #print(loss)
        return loss


    def compute_critic_loss(self, observation_batch, reward_batch, next_observation_batch, done_batch):
        # minimize mean((r + gamma * V(s_t1) - V(s_t))^2)
        value = self._critic(observation_batch).squeeze().float()#.squeeze(0)
        done_batch=torch.tensor(done_batch)#.view(-1, 1)
        reward_batch=torch.tensor(reward_batch)#.view(-1, 1).float()
        target = reward_batch + ~done_batch * self._gamma * self._critic(next_observation_batch).detach().squeeze()
        target=target.float()
        
        loss = F.mse_loss(value, target)
        #print(loss)
        return loss


    def update_actor(self, observation_batch, action_batch, reward_batch, next_observation_batch, done_batch):
        # compute the advantages using the critic and update the actor parameters
        value = self._critic(observation_batch).detach().squeeze()
        #print(value.shape)
        done_batch=torch.tensor(done_batch)#.view(-1, 1)
        #print(done_batch.shape)
        reward_batch=torch.tensor(reward_batch)#.view(-1, 1).float()
        #print(reward_batch.shape)
        target = reward_batch + ~done_batch * self._gamma * self._critic(next_observation_batch).detach().squeeze()#[0]
        #print(target.shape)
        #print(value.shape)
        advantage = (target - value).detach()

        # use self._compute_actor_loss
        loss=-self._compute_actor_loss(observation_batch, action_batch,advantage)
        #print(loss)
        # Backpropagation
        self._actor.zero_grad()
        loss.backward()
        self._actor_optimizer.step()
        
        
        
    def update_critic(self, observation_batch, reward_batch, next_observation_batch, done_batch):
        # update the critic
        # use self.compute_critic_loss
        loss=self. compute_critic_loss(observation_batch, reward_batch, next_observation_batch, done_batch).float()
        #print(loss)
        # Backpropagation
        self._critic.zero_grad()
        loss.backward()
        self._critic_optimizer.step()
        