import numpy as np


class Buffer:

    def __init__(self, dim_states, dim_actions, max_size, sample_size):

        assert sample_size < max_size, "Sample size cannot be greater than buffer size"
        
        self._buffer_idx     = 0
        self._exps_stored    = 0
        self._buffer_size    = max_size
        self._sample_size    = sample_size

        self._s_t_array      = np.zeros((max_size, dim_states))
        self._a_t_array      = np.zeros((max_size))
        self._s_t1_array     = np.zeros((max_size, dim_states))


    def store_transition(self, s_t, a_t, s_t1):
        
        # Add transition to the buffer
        self._s_t_array[self._buffer_idx]=s_t   
        self._a_t_array[self._buffer_idx]=a_t  
        self._s_t1_array[self._buffer_idx]=s_t1 

        # Aumento de indice y reinicio de indice si superamos capacidad
        self._buffer_idx = (self._buffer_idx + 1) % self._buffer_size
        self._exps_stored += 1
        #print(self._exps_stored)
        #pass

    
    def get_batches(self):
        #print(self._exps_stored)
        assert self._exps_stored + 1 > self._sample_size, "Not enough samples has been stored to start sampling"

        # Get all the data contained in the buffer as batches 
        batches_s_t = [self._s_t_array[i:i+self._sample_size] for i in range(0, len(self._s_t_array), self._sample_size)]
        batches_a_t = [self._a_t_array[i:i+self._sample_size] for i in range(0, len(self._a_t_array), self._sample_size)]
        batches_s_t1 = [self._s_t1_array[i:i+self._sample_size] for i in range(0, len(self._s_t1_array), self._sample_size)]

        return [batches_s_t,batches_a_t,batches_s_t1]
            