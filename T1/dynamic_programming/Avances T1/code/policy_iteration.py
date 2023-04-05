import numpy as np
import matplotlib.pyplot as plt

from grid_world import GridWorld
from utils import display_policy
from utils import display_value_function
from utils import new_state,Reward,ortogonal_movements


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

        # Inicializar delta
        delta=0

        num_update_value_function=0

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

                        # Politica estado actual
                        old_action=self._policy[actual_state]

                        reward=Reward(actual_state,self._reward_grid)

                        # Movimientos ortogonales
                        x,y=ortogonal_movements(old_action)

                        # Estado dir
                        state_dir=new_state(actual_state,old_action,self._reward_grid)

                        # estado ortogonal 1
                        state_x=new_state(actual_state,x,self._reward_grid)

                        # Estado ortogonal 2
                        state_y=new_state(actual_state,y,self._reward_grid)

                        # Value functions next potential states: Vs'
                        Vs_dir=self._value_function[state_dir]
                        Vs_x=self._value_function[state_x]
                        Vs_y=self._value_function[state_y]
                        
                        new_value=p_dir*(reward+gamma*Vs_dir)+p_random*0.5*(reward+gamma*Vs_x)+p_random*0.5*(reward+gamma*Vs_y)

                        self._value_function[actual_state]=new_value

                        delta=max(delta,abs(old_value-new_value))

                        #print(delta)
            num_update_value_function+=1
            if delta < v_thresh:
                print("# de actualizaciones a función de valor: ",num_update_value_function)
                break
                            
    def _policy_improvement(self, nb_iters, p_dir, gamma):

        # Policy improvement
        # Code your algorithm here (P1-2) (you can add auxiliary functions if needed)
        p_random    = 1 - p_dir
        value_rows, value_cols = self._value_function.shape
        stable_policy = True
        
        for j in range(1, value_rows - 1):
            for i in range(1, value_cols - 1):
            
                if self._reward_grid[(j,i)]==self._cell_value:
                    
                    # Estado inicial
                    actual_state=(j,i)

                    # Politica estado actual
                    old_action=self._policy[actual_state]

                    reward=Reward(actual_state,self._reward_grid)

                    list_values=[]
                    for action in range(4):

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
            
                    new_action=np.array(list_values).argmax()
                    
                    # Update nueva mejor acción
                    self._policy[actual_state]=new_action

                    if new_action!=old_action:
                            stable_policy=False

                
        return stable_policy


    def run_policy_iteration(self, p_dir, nb_iters, gamma, v_thresh):
        stable_policy = False
        policy_count=0
        while not stable_policy:
            self._policy_evaluation(nb_iters, p_dir, gamma, v_thresh)
            policy_count+=1
            stable_policy = self._policy_improvement(nb_iters, p_dir, gamma)   
        print("# Llamados a función policy evaluation: ", policy_count)     
        

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
