import numpy as np

class ReplayBuffer:

    def __init__(self, dim_states, dim_actions, max_size, sample_size):

        assert sample_size < max_size, "Sample size cannot be greater than buffer size"
        
        self._buffer_idx     = 0
        self._exps_stored    = 0
        self._buffer_size    = max_size
        self._sample_size    = sample_size

        self._s_t_array      = None
        self._a_t_array      = None
        self._r_t_array      = None
        self._s_t1_array     = None
        self._term_t_array   = None


    def store_transition(self, s_t, a_t, r_t, s_t1, done_t):

        # Add transition to replay buffer according to self._buffer_idx

        # Update replay buffer index
        self._buffer_idx = None
        self._exps_stored += 1
    

    def sample_transitions(self):
        assert self._exps_stored + 1 > self._sample_size, "Not enough samples have been stored to start sampling"
        
        sample_idxs = None
        """
        return (self._s_t_array[sample_idxs],
                self._a_t_array[sample_idxs],
                self._r_t_array[sample_idxs],
                self._s_t1_array[sample_idxs],
                self._term_t_array[sample_idxs])
        """
