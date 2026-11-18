import numpy as np
    
def add_poission_noise(data_in, fac):
    '''
        DESCRIPTION: 
            this function transforms a given data into countstatistcs with poisson statistics
            Draw exact Poisson photon counts with mean data_in.
            This function requires photon-counting data and unit gain (fac=1).

        CALL: 
            - data_in: input noise-free simulated data
            - fac: must be 1; detector gain is not modeled here
            - data_out: output noisy data
            - data_noise: added Poission noise data

        UPDATES:
        2021/11/01 (Chengpeng Wu): first version
    ''' 
    
    if fac != 1:
        raise ValueError('Exact Poisson counting requires fac=1; define gain separately')
    if not np.all(np.isfinite(data_in)) or np.any(data_in < 0):
        raise ValueError('Poisson means must be finite and nonnegative')
    return np.random.poisson(data_in).astype(float)
