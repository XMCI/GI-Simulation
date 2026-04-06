import numpy as np
    
def calc_FCA(psc, nPeriods, dim):
    '''
        DESCRIPTION:
            CALC_FCA perform Fourier analysis on phase stepping curve to retrieve the three parameters
        CALL:
            -psc: phase stepping curve, flat or sample
            -nPeriods: number of periods in phase stepping
        OUTPUT:
            -amp: amplitude of the PSC
            -vis: visibility
            -phi: phase of the PSC
        UPDATES:
            2024/11/27 (Longchao Men): add '2D' mode
            2022/09/08 (Peiyuan Guo): first version
    '''
    if dim == '1D':
        nSteps = np.size(psc, 1) # the shape of psc is (npixels, nSteps)
        fft_psc = np.fft.fft(psc, axis=1)
        c0 = np.abs(fft_psc[:, 0]) / nSteps
        c1 = np.abs(fft_psc[:, nPeriods]) / nSteps
        vis = 2 * c1 / c0
        amp = c0
        phi = np.angle(fft_psc[:, nPeriods])
    
    elif dim == '2D':
        nSteps = np.size(psc, 0) # the shape of psc is (nSteps, npixels, npixels)
        fft_psc = np.fft.fft(psc, axis=0)
        c0 = np.abs(fft_psc[0,:,:]) / nSteps
        c1 = np.abs(fft_psc[nPeriods,:,:]) / nSteps
        vis = 2 * c1 / c0
        amp = c0
        phi = np.angle(fft_psc[nPeriods,:,:])

    return amp,vis,phi
    