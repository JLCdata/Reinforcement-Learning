import numpy as np


class QLearningAgent():

    def __init__(self, states_high_bound, states_low_bound, nb_actions, nb_episodes, gamma, alpha, epsilon):
    
        self._epsilon = epsilon
        self._gamma = gamma
        self._alpha = alpha

        self._states_high_bound = states_high_bound
        self._states_low_bound = states_low_bound
        self._nb_actions = nb_actions
        
        # Define these variables (P2-2)
        self._nb_states = None
        self._tabular_q = None
        

    """ Epsilon-greedy policy 
    """
    def select_action(self, observation, greedy=False):
        # P1-3
        if np.random.random() > self._epsilon or greedy:
            pass

        else:
            pass
        
        action = 2
        return action


    """ Q-function update
    """
    def update(self, ob_t, ob_t1, action, reward, is_done):
        # P1-3
        terminal_condition = ob_t1[0] > 0.5

        if is_done and terminal_condition:
           pass

        else:
            pass
        
        # P1-5 only
        if is_done and self._epsilon > 0.0:
            pass
