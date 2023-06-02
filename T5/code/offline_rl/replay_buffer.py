import numpy as np

class ReplayBuffer:

    def __init__(self, dim_states, dim_actions, max_size, sample_size):

        assert sample_size < max_size, "Sample size cannot be greater than buffer size"
        
        self._buffer_idx     = 0
        self._exps_stored    = 0
        self._buffer_size    = max_size
        self._sample_size    = sample_size

        self._s_t_array      = np.empty(shape=(self._buffer_size, dim_states), dtype=np.float32)
        self._a_t_array      = np.empty(shape=(self._buffer_size, ), dtype=np.float32)
        self._r_t_array      = np.empty(shape=(self._buffer_size, ), dtype=np.float32)
        self._s_t1_array     = np.empty(shape=(self._buffer_size, dim_states), dtype=np.float32)
        self._term_t_array   = np.empty(shape=(self._buffer_size, ), dtype=np.float32)


    def store_transition(self, s_t, a_t, r_t, s_t1, done_t):

        # Add transition to replay buffer
        self._s_t_array[self._buffer_idx]     = s_t
        self._a_t_array[self._buffer_idx]     = a_t
        self._r_t_array[self._buffer_idx]     = r_t
        self._s_t1_array[self._buffer_idx]    = s_t1
        self._term_t_array[self._buffer_idx]  = float(done_t)

        # Update replay buffer index
        self._buffer_idx = (self._buffer_idx + 1) % self._buffer_size
        self._exps_stored += 1
    

    def sample_transitions(self):
        assert self._exps_stored + 1 > self._sample_size, "Not enough samples has been stored to start sampling"

        sample_idxs = np.random.randint(low=0, high=min(self._exps_stored, self._buffer_size), size=self._sample_size)

        return (self._s_t_array[sample_idxs],
                self._a_t_array[sample_idxs],
                self._r_t_array[sample_idxs],
                self._s_t1_array[sample_idxs],
                self._term_t_array[sample_idxs])
