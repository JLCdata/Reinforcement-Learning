import numpy as np


class QLearningAgent():
    
    def __init__(self, states_high_bound, states_low_bound, nb_actions, nb_episodes, gamma, alpha, epsilon):
    
        self._epsilon = epsilon
        self._gamma = gamma
        self._alpha = alpha

        self._states_high_bound = states_high_bound
        self._states_low_bound = states_low_bound
        self._nb_actions = nb_actions
        
        # (P1-5)
        self._nb_episodes=nb_episodes

        # Define these variables (P1-2)
        # Q-value function
        self._nb_states = 25
        self._tabular_q =np.zeros((self._nb_states,self._nb_states,self._nb_actions))
        

    """ Discretización de estados
    """
    def discretization_states(self, observation):

        # Discretización del ambiente 
        bins=np.array([self._nb_states,self._nb_states])
        low=self._states_low_bound
        high=self._states_high_bound
        buckets=[np.linspace(l,h,b) for l,h,b in zip(low,high,bins)]
        discrete_observation=tuple(np.digitize(cont,buck) for cont,buck in zip(observation,buckets))

        return discrete_observation

    """ Epsilon-greedy policy 
    """
    def select_action(self, observation, greedy=False):

        discrete_observation=self.discretization_states(observation)
        
        # P1-3
        if np.random.random() > self._epsilon or greedy:
            
            # Action values
            av = self._tabular_q[discrete_observation]

            # Action con mayor q-value
            action=np.random.choice(np.flatnonzero(av == av.max()))
     
        else:
            # Exploración
            action=np.random.randint(3)
        
        return action


    """ Q-function update
    """
    def update(self, ob_t, ob_t1, action, reward, is_done):
        
        # P1-3
        terminal_condition = ob_t1[0] > 0.5
        
        # Discretization
        discretization_ob_t=self.discretization_states(ob_t)
        discretization_ob_t1=self.discretization_states(ob_t1)

        if is_done and terminal_condition:
           # Condición de termino
           reward=0
           
        else:
            # Se mantiene el reward
            reward=reward

        # Next_action greedy
        next_action=self.select_action(ob_t1,greedy=True)

        # Q-value antiguo 
        qsa=self._tabular_q[discretization_ob_t][action]

        # Q-value nuevo
        next_qsa=self._tabular_q[discretization_ob_t1][next_action]

        # Update Q-function
        self._tabular_q[discretization_ob_t][action]=qsa+self._alpha*(reward+self._gamma*next_qsa-qsa)

        # P1-5 only
        if is_done and self._epsilon > 0.0:
            pass
            #self._epsilon-=self._epsilon/self._nb_episodes
            