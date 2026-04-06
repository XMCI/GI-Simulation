import numpy as np
    
def add_poission_noise(data_in, fac):
    '''
        DESCRIPTION: 
            this function transforms a given data into countstatistcs with poisson statistics
            each new value is mynrand with mu = value; sigma = fac*sqrt(value).
            Quantization noise is included by rounding the data.

        CALL: 
            - data_in: input noise-free simulated data
            - fac: proportionality factor depending on detector properties
            - data_out: output noisy data
            - data_noise: added Poission noise data

        UPDATES:
        2021/11/01 (Chengpeng Wu): first version
    ''' 
    
    data_noise = np.round(np.multiply(fac*np.sqrt(data_in), np.random.randn(*data_in.shape)))    
    
    data_out = data_in + data_noise
    return data_out
    