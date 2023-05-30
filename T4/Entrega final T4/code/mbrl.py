import torch
import torch.nn as nn

import numpy as np

from buffer import Buffer

import matplotlib.pyplot as plt

from torch.optim import AdamW

import torch.nn.functional as F

class Model(nn.Module):
    
    def __init__(self, dim_states, dim_actions, continuous_control):
        super(Model, self).__init__()
        
        self._fc1 = nn.Sequential(
        nn.Linear(dim_states+1, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, dim_states)
        )
        self.continuous_control=continuous_control
       
    def forward(self, state, action):

        if len(state.shape)>1:

            concat_o_a=np.concatenate((state,action.reshape(-1,1)),axis=1)
            input=torch.from_numpy(concat_o_a).float()
            #print(input)
            output=self._fc1(input)
        
        else:
            
            action=np.array(action if self.continuous_control else [action])
            #print(action)
            #print(state)
            concat_o_a=np.concatenate((state,action))
            input=torch.from_numpy(concat_o_a).float()
            #print(input)
            output=self._fc1(input)

        return output


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
            random_actions = np.array([[np.array([np.random.random()*4-2]).astype("float32") for i in range(self._planning_horizon)] for j in range(self._nb_trajectories)])
            
        else:
            random_actions = np.array([[np.random.randint(2) for i in range(self._planning_horizon)] for j in range(self._nb_trajectories)])
            

        # Construct initial observation 
        o_t = np.repeat(observation,self._nb_trajectories).reshape(-1,self._dim_states)
        #print(o_t)
        rewards = np.zeros((self._nb_trajectories, ))
        
        for i in range(self._planning_horizon):
            #print(rewards.shape)

            # Get a_t
            a_t=random_actions[:,i].squeeze()

            # Predict next observation using the model
            o_t1=self._model(o_t,a_t).detach().numpy()

            # Compute reward (use reward_function)
            #print(rewards)
            
            rewards=rewards+self._reward_function(o_t,a_t)
            
            # Update
            o_t = o_t1
        
        # Return the best sequence of actions
        index_best_actions=np.argmax(rewards)
        #print(o_t1.shape)
        #print(o_t.shape)
        #print(a_t.shape)
        #print(rewards.shape)
        #print(random_actions[index_best_actions])
        return random_actions[index_best_actions]


class MBRLAgent:

    def __init__(self, dim_states, dim_actions, continuous_control, model_lr, buffer_size, batch_size, 
                       planning_horizon, nb_trajectories, reward_function):

        self._dim_states = dim_states
        self._dim_actions = dim_actions

        self._continuous_control = continuous_control

        self._model_lr = model_lr

        self._model = Model(self._dim_states, self._dim_actions, self._continuous_control)

        # Adam optimizer
        self._model_optimizer = AdamW(self._model.parameters(), lr=self._model_lr)

        self._buffer = Buffer(self._dim_states, self._dim_actions, buffer_size, batch_size)
        
        self._planner = RSPlanner(self._dim_states, self._dim_actions, self._continuous_control, 
                                  self._model, planning_horizon, nb_trajectories, reward_function)


    def select_action(self, observation, random=False):

        if random:
            # Return random action
            if self._continuous_control:
                return np.array([np.random.random()*4-2]).astype("float32")
            
            return np.random.randint(2)

        # Generate plan
        plan = self._planner.generate_plan(observation)

        # Return the first action of the plan
        if self._continuous_control:
            return plan[0]
        
        return plan[0]


    def store_transition(self, s_t, a_t, s_t1):
        self._buffer.store_transition(s_t,a_t,s_t1)


    def update_model(self):
        
        s_t,a_t,s_t1=self._buffer.get_batches()

        #list_loss=[]
        for x,y,z in zip(s_t,a_t,s_t1):
            
            # Use the batches to train the model
            # loss: avg((s_t1 - model(s_t, a_t))^2)
            #loss=((self._model(x,y)-torch.tensor(z))**2).mean()

            loss=F.mse_loss(self._model(x,y).float(), torch.tensor(z).float())
            self._model.zero_grad()
            loss.backward()
            self._model_optimizer.step()
            #list_loss.append(loss.item())
        
        #epoch=len(list_loss)
        #epocas=[i for i in range(epoch)]  # Lista con épocas hasta el ultimo Check Point para poder graficar
#
        #plt.plot(epocas,list_loss) # Plot entrenamiento
#
        #plt.legend(["Loss Entrenamiento"], loc ="upper right")
        #plt.title('Curvas Loss')
        #plt.xlabel('# Epoch')
        #plt.ylabel('Loss')
        #plt.show()
        