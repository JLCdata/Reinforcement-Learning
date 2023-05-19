import numpy as np

class Buffer:

    def __init__(self, dim_states, dim_actions, max_size, sample_size):

        assert sample_size < max_size, "Sample size cannot be greater than buffer size"
        
        self._buffer_idx     = 0
        self._exps_stored    = 0
        self._buffer_size    = max_size
        self._sample_size    = sample_size

        self._s_t_array      = None
        self._a_t_array      = None
        self._s_t1_array     = None


    def store_transition(self, s_t, a_t, s_t1):
        # Add transition to the buffer
        pass

    
    def get_batches(self):
        assert self._exps_stored + 1 > self._sample_size, "Not enough samples has been stored to start sampling"
        # Get all the data contained in the buffer as batches 
        pass
            