import numpy as np
import matplotlib.pyplot as plt

from grid_world import GridWorld
from utils import display_policy
from utils import display_value_function
from utils import new_state,Reward,ortogonal_movements

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

        # V(s) = max_a sum_s'(P_ss'^a[R_ss'^a+gamma*V(s')]) 
        # We perform all posible actions, check the value, and update according to the max value found
        
        num_update_value_function=0

        # Inicializar delta
        delta=0
        
        for _ in range(nb_iters):

            # Inicializar delta
            delta=0
            
            # Indexes for skipping external walls (you may change them)
            for j in range(1, value_rows - 1):
                for i in range(1, value_cols - 1):

                    if self._reward_grid[(j,i)]==self._cell_value:
                        
                        # Estado inicial
                        actual_state=(j,i)

                        # Función de valor Vs antiguo
                        old_value=self._value_function[actual_state]

                        # Max function value to an specific action
                        list_values=[]
                        for action in range(4):

                            # Reward
                            reward=Reward(actual_state,self._reward_grid)

                            # Movimientos ortogonales
                            x,y=ortogonal_movements(action)

                            # Estado dir
                            state_dir=new_state(actual_state,action,self._reward_grid)

                            # estado ortogonal 1
                            state_x=new_state(actual_state,x,self._reward_grid)

                            # Estado ortogonal 2
                            state_y=new_state(actual_state,y,self._reward_grid)

                            # Value functions next potential states: Vs'
                            Vs_dir=self._value_function[state_dir]
                            Vs_x=self._value_function[state_x]
                            Vs_y=self._value_function[state_y]
                            
                            new_value=p_dir*(reward+gamma*Vs_dir)+p_random*0.5*(reward+gamma*Vs_x)+p_random*0.5*(reward+gamma*Vs_y)

                            list_values.append(new_value)
                        
                        # Best action
                        new_action=np.array(list_values).argmax()

                        # Max value
                        new_max_value=np.array(list_values).max()

                        # Update nueva mejor acción
                        self._policy[actual_state]=new_action

                        # Update max value
                        self._value_function[actual_state]=new_max_value
                    
                        # delta value function
                        delta=max(delta,abs(old_value-new_max_value))
                        #print(delta)
                          
            num_update_value_function+=1
          
            if delta < v_thresh:
                print("# de actualizaciones a función de valor: ",num_update_value_function)
                break


if __name__ == '__main__':
    '''
    # P2-2
    world = GridWorld(height=14, width=16)

    value_iterator = ValueIterator(reward_grid=world._rewards,
                                   wall_value=None,
                                   cell_value=-1,
                                   terminal_value=0)

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
    '''

    '''
    # P2-3
    world = GridWorld(height=14, width=16)

    value_iterator = ValueIterator(reward_grid=world._rewards,
                                   wall_value=None,
                                   cell_value=-1,
                                   terminal_value=0)

    value_iterator.run_value_iteration(p_dir=1,
                                       nb_iters=1000,
                                       gamma=0.9,
                                       v_thresh=0.0001)

    world.display()

    display_value_function(value_iterator._value_function)

    display_policy(world._grid,
                   value_iterator._reward_grid,
                   value_iterator._policy)

    plt.show()
    '''

    '''
    # P2-4 i
    world = GridWorld(height=14, width=16)

    value_iterator = ValueIterator(reward_grid=world._rewards,
                                   wall_value=None,
                                   cell_value=-1,
                                   terminal_value=0)

    value_iterator.run_value_iteration(p_dir=0.6,
                                       nb_iters=1000,
                                       gamma=0.2,
                                       v_thresh=0.0001)

    world.display()

    display_value_function(value_iterator._value_function)

    display_policy(world._grid,
                   value_iterator._reward_grid,
                   value_iterator._policy)

    plt.show()

    '''
    # P2-4 ii
    world = GridWorld(height=14, width=16)

    value_iterator = ValueIterator(reward_grid=world._rewards,
                                   wall_value=None,
                                   cell_value=-1,
                                   terminal_value=0)

    value_iterator.run_value_iteration(p_dir=0.6,
                                       nb_iters=1000,
                                       gamma=1,
                                       v_thresh=0.0001)

    world.display()

    display_value_function(value_iterator._value_function)

    display_policy(world._grid,
                   value_iterator._reward_grid,
                   value_iterator._policy)

    plt.show()
    
    
