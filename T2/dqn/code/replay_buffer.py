import numpy as np

class ReplayBuffer:
    
    def __init__(self, dim_states, dim_actions, max_size, sample_size):

        assert sample_size < max_size, "Sample size cannot be greater than buffer size"
        
        self._buffer_idx     = 0
        self._exps_stored    = 0
        self._buffer_size    = max_size
        self._sample_size    = sample_size

        self._s_t_array      = np.zeros((max_size, dim_states))
        self._a_t_array      = np.zeros((max_size))
        self._r_t_array      = np.zeros((max_size,))
        self._s_t1_array     = np.zeros((max_size, dim_states))
        self._term_t_array   = np.zeros((max_size,))


    def store_transition(self, s_t, a_t, r_t, s_t1, done_t):

        # Add transition to replay buffer according to self._buffer_idx
        self._s_t_array[self._buffer_idx]=s_t   
        self._a_t_array[self._buffer_idx]=a_t  
        self._r_t_array[self._buffer_idx]=r_t  
        self._s_t1_array[self._buffer_idx]=s_t1 
        self._term_t_array[self._buffer_idx]=done_t

        # Update replay buffer index
        # Aumento de indice y reinicio de indice si superamos capacidad
        self._buffer_idx = (self._buffer_idx + 1) % self._buffer_size
        self._exps_stored += 1

    
    def sample_transitions(self):
        assert self._exps_stored + 1 > self._sample_size, "Not enough samples have been stored to start sampling"
        
        sample_idxs = np.random.choice(self._buffer_size, size=self._sample_size,replace=False)
        
        return (self._s_t_array[sample_idxs],
                self._a_t_array[sample_idxs],
                self._r_t_array[sample_idxs],
                self._s_t1_array[sample_idxs],
                self._term_t_array[sample_idxs])
        
