import numpy as np
import matplotlib.pyplot as plt

from grid_world import GridWorld
from utils import display_policy
from utils import display_value_function


class ValueIterator():

    def __init__(self, reward_grid, wall_value, cell_value, terminal_value):

        self._reward_grid = reward_grid
        self._wall_value = wall_value
        self._cell_value = cell_value
        self._terminal_value = terminal_value

        self._value_function = np.zeros(self._reward_grid.shape)
        self._value_function *= self._reward_grid
        self._policy = self._value_function.copy()


    def run_value_iteration(self, p_dir, nb_iters, gamma, v_thresh):
        p_random    = 1 - p_dir
        #p_side
        value_rows, value_cols = self._value_function.shape

        # Code your algorithm here (P2-1) (you can add auxiliary functions if needed)
        # Notice that in the reward grid walls are nans, traversable cells are -1's and the goal is 0.
        
        # V(s) = max_a sum_s'(P_ss'^a[R_ss'^a+gamma*V(s')]) 
        # We perform all posible actions, check the value, and update according to the max value found

        for _ in range(nb_iters):

            # Indexes for skipping external walls (you may change them)
            for j in range(1, value_rows - 1):
                for i in range(1, value_cols - 1):

                    pass
            
            """
            if something < v_thresh:
                break
            """


if __name__ == '__main__':

    world = GridWorld(height=14, width=16)

    value_iterator = ValueIterator(reward_grid=world._rewards,
                                   wall_value=None,
                                   cell_value=-1,
                                   terminal_value=0)

    # Default parameters for P2-2 (change them for P2-3 & P2-4 & P2-5)
    value_iterator.run_value_iteration(p_dir=0.8,
                                       nb_iters=1000,
                                       gamma=0.9,
                                       v_thresh=0.0001)

    world.display()

    display_value_function(value_iterator._value_function)

    display_policy(world._grid,
                   value_iterator._reward_grid,
                   value_iterator._policy)

    plt.show()
