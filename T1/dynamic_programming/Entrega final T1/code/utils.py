import numpy as np
import matplotlib.pyplot as plt

def display_value_function(value_function):

    fig, ax = plt.subplots()
    value_rows, value_cols = value_function.shape
    value_function_display = value_function.copy()
    value_function_display = np.nan_to_num(value_function_display)
    value_function_display[np.isnan(value_function)] = np.min(value_function_display)
    threshold = (np.max(value_function_display) - np.min(value_function_display)) / 2
    
    for j in range(value_rows):
        for i in range(value_cols):
            if not np.isnan(value_function[j , i]):
                ax.text(i, j, format(value_function[j, i], '.1f'), ha='center', va='center', 
                        color='white' if abs(value_function[j, i]) > threshold else 'black')

    ax.imshow(value_function_display, cmap='gray')

    plt.title('Value Function')
    plt.axis('off')
    fig.tight_layout()

    plt.savefig('value_function.pdf')

    
def display_policy(world_grid, reward_grid, policy):

    fig, ax = plt.subplots()
    rows, cols = reward_grid.shape
    
    arrow_symols = [u'\u2191', u'\u2193',u'\u2192', u'\u2190']

    for j in range(rows):
        for i in range(cols):
            if reward_grid[j, i] == 0.0:
                ax.text(i, j, 'G', ha='center', va='center')

            elif not np.isnan(policy[j , i]):
                ax.text(i, j, arrow_symols[int(policy[j, i])], ha='center', va='center')

    ax.imshow(world_grid, cmap='gray')

    plt.title('Policy')
    plt.axis('off')
    fig.tight_layout()

    plt.savefig('policy.pdf')


### New auxiliar functions ####

# Función para sumar arreglos (x,y)
def sum_array(arreglo1, arreglo2):
    resultado = ()
    for elemento1, elemento2 in zip(arreglo1, arreglo2):
        suma = elemento1 + elemento2
        resultado += (suma,)
    return resultado

# Función para definir el siguiente estado dado la acción
def new_state(initial_state,action,reward_grid):

     # 0: subir, 1: bajar
    if (action==0) or (action==1):
    
        # Según la acción se define el delta state que permite obtener el nuevo potencial estado
        delta_state=(-1,0) if action==0 else (1,0)

        # Se obtiene el nuevo potencial estado mediante suma vectorial
        potential_new_state=sum_array(initial_state, delta_state)

    # 2: derecha,3: izquierda
    if (action==2) or (action==3):

        # Según la acción se define el delta state que permite obtener el nuevo potencial estado
        delta_state=(0,1) if  action==2  else  (0,-1)

        # Se obtiene el nuevo potencial estado mediante suma vectorial 
        potential_new_state=sum_array(initial_state, delta_state)

    # Permanecemos en el estado inicial si es que no se puede recorrer dicho estado sino, vamos al nuevo estado
    final_state=potential_new_state if ~np.isnan(reward_grid[potential_new_state]) else initial_state

    return final_state

# Función que retorna el reward asociado al estado actual
def Reward(initial_state,reward_grid):
    
    return reward_grid[initial_state] 

# Entrega acciones ortogonales a la entregada
def ortogonal_movements(action):

     if (action==0) or (action==1):
          
          # vertical 1
          x=2
          # vertical 2
          y=3
          return (x,y)
      
     if (action==2) or (action==3):
          
          # vertical 1
          x=0
          # vertical 2
          y=1
          return (x,y)     

### New auxiliar functions ####