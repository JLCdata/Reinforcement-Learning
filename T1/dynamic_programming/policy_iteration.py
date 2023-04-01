import numpy as np
import matplotlib.pyplot as plt

from grid_world import GridWorld
from utils import display_policy
from utils import display_value_function

class PolicyIterator():

    def __init__(self, reward_grid, wall_value, cell_value, terminal_value):

        self._reward_grid = reward_grid
        self._wall_value = wall_value
        self._cell_value = cell_value
        self._terminal_value = terminal_value

        self._value_function = np.zeros(self._reward_grid.shape)
        self._value_function *= self._reward_grid
        self._policy = self._value_function.copy()


    def _policy_evaluation(self, nb_iters, p_dir, gamma, v_thresh):
        # Policy evaluation
        # Code your algorithm here (P1-2) (you can add auxiliary functions if needed)
        
        p_random    = 1 - p_dir
        #p_side
        value_rows, value_cols = self._value_function.shape

        for _ in range(nb_iters):

            # Indexes for skipping external walls (you may change them)
            for j in range(1, value_rows - 1):
                for i in range(1, value_cols - 1):

                    pass
            
            """
            if something < v_thresh:
                break
            """
                    
    def _policy_improvement(self, nb_iters, p_dir, gamma):
        # Policy improvement
        # Code your algorithm here (P1-2) (you can add auxiliary functions if needed)
        
        p_random    = 1 - p_dir
        value_rows, value_cols = self._value_function.shape

        old_policy = self._policy.copy()
        stable_policy = True

        for j in range(1, value_rows - 1):
            for i in range(1, value_cols - 1):
                
                """if something:
                    stable_policy = False
                """
        
        return stable_policy


    def run_policy_iteration(self, p_dir, nb_iters, gamma, v_thresh):
        stable_policy = False

        while not stable_policy:
            self._policy_evaluation(nb_iters, p_dir, gamma, v_thresh)
            stable_policy = self._policy_improvement(nb_iters, p_dir, gamma)        
        

if __name__ == '__main__':

    world = GridWorld(height=14, width=16)
    policy_iterator = PolicyIterator(reward_grid=world._rewards,
                                     wall_value=None,
                                     cell_value=-1,
                                     terminal_value=0)

    # Default parameters for P1-3 (change them for P2-3)
    policy_iterator.run_policy_iteration(p_dir=0.8,
                                         nb_iters=1000,
                                         gamma=0.9,
                                         v_thresh=0.0001)

    world.display()

    display_value_function(policy_iterator._value_function)

    display_policy(world._grid,
                   policy_iterator._reward_grid,
                   policy_iterator._policy)

    plt.show()
