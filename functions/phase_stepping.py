import numpy as np
import sys
# from functions.create_grating import create_grating
from functions.create_grating_v2 import create_grating
from functions.detector import detector
from functions.add_poission_noise import add_poission_noise
import cupy as cp
    
def phase_stepping(I1, nSteps, nPeriods, E, coord, G2, ER, pixel_size, chi, dim, gpu_flag, noise_flag, logger, det_type='PhotonCounting'):
    '''
        DESCRIPTION: 
            this function performs the phase stepping and the detection of the signal. 
            Optional Gaussian (EnergyIntegral) or Poisson (PhotonCounting) noise.
        
        CALL: 
            - I1: incident photon intensity before G2
                - 1D: shape (nE, nX)
                - 2D: shape (nE, nX, nY)
            - nSteps: number of phase steps
            - nPeriods: number of periods in phase-stepping
            - E: vector containing the different energy bands
            - x: spatial coordinates
            - G2: the G2 grating struct
                - G2.period: pitch of grating
                - G2.dc: duty cycle of grating
                - G2.material: plated material in the grating surface
                - G2.thickness: the thickness of plated material
                - G2.base_material: base material of the grating, default 'air'
                - G2.base_thickness: the thickness of the base material, default 0
            - ER: detector response
            - pixel_size: detector pixel size
            - nbits: number of bits
            - chi: Gaussian standard deviation in PSC units for EnergyIntegral;
                   must be 1 for PhotonCounting
            - gpu_flag: logical variable whether using the GPU or not
            - noise_flag: add noise if 1
            - det_type: EnergyIntegral or PhotonCounting (default)
        OUTPUT:
            - PSC: phase stepping curve
            
        UPDATES:
        2024/11/27 (Longchao Men): add the 2D mode
        2021/11/12 (Chengpeng Wu): add the gpu_flag and corresponding operations
        2021/11/09 (Chengpeng Wu): create G2 and I1 before phase-stepping to avoid repeated calculation
        2021/11/01 (Chengpeng Wu): first version
    '''
    
    for name, value in [('nSteps', nSteps), ('nPeriods', nPeriods)]:
        if not np.isfinite(value) or int(value) != value or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    nSteps, nPeriods = int(nSteps), int(nPeriods)
    dx = coord.x[0, 1] - coord.x[0, 0]
    FOV = coord.x.shape[1] * dx
    samples_per_period = G2.period / dx
    samples_per_step = samples_per_period / nSteps
    if not np.isfinite(samples_per_period) or samples_per_period <= 0:
        raise ValueError('G2 period and sampling interval must be positive and finite')
    if not np.isclose(samples_per_period, round(samples_per_period), rtol=1e-9, atol=1e-9):
        raise ValueError('G2 period must be an integer number of samples')
    if not np.isclose(samples_per_step, round(samples_per_step), rtol=1e-9, atol=1e-9):
        raise ValueError('Phase step must be an integer number of samples for np.roll')
    # Use the same integer pixel count as detector(), not arange/ceil.
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError('Invalid detector pixel size')
    pixel_ratio = FOV / pixel_size
    if not np.isfinite(pixel_ratio) or pixel_ratio < 1:
        raise ValueError('Invalid detector field of view')
    n_pixels = int(round(pixel_ratio))
    if not np.isclose(pixel_ratio, n_pixels, rtol=1e-9, atol=1e-9):
        raise ValueError('FOV must contain an integer number of detector pixels')
    if any(n % n_pixels != 0 for n in I1.shape[1:]):
        raise ValueError('Each detector pixel must contain an integer sample block')
    
    if dim == '1D':
        # vector containing the phase step distances
        s = np.linspace(0, G2.period*nPeriods, nSteps*nPeriods+1)
        s = s[:-1]

        PSC = np.zeros((n_pixels, np.size(s)))
        g20 = create_grating(G2, E, coord, dim, gpu_flag, logger, abs_flag=1)

        for ii in np.arange(0, np.size(s)):
            shift_x = np.round(s[ii] / dx).astype(int)
            g2 = np.roll(g20, shift_x, axis=1)
            PSC[:,ii] = detector(I1 * g2, ER, FOV, pixel_size, g2, dim) # shape: (n_pixels, nSteps)

            print("\r", end="")
            print("Phase stepping: {}% ".format(round((ii+1)/len(np.arange(0, np.size(s)))*100), 2), end="")
            sys.stdout.flush()
        print()

        # Add detector noise after pixel integration and energy summation.
        if noise_flag == 1:
            if det_type == 'EnergyIntegral':
                if not np.isfinite(chi) or chi < 0:
                    raise ValueError('Gaussian noise std chi must be finite and nonnegative')
                PSC = PSC + np.random.normal(0.0, chi, size=PSC.shape)
            else:
                PSC = add_poission_noise(PSC, chi)
        
    elif dim == '2D':
        # vector containing the phase step distances
        s = np.linspace(0, G2.period*nPeriods, nSteps*nPeriods+1)
        s = s[:-1]

        PSC = np.zeros((np.size(s), n_pixels, n_pixels))
        g20 = create_grating(G2, E, coord, dim, gpu_flag, logger, abs_flag=1) #abs_flag is used to create the absolute value of the grating
        if gpu_flag:
            g20 = cp.asarray(g20)

        for ii in np.arange(0, np.size(s)):
            shift_x = np.round(s[ii] / dx).astype(int)

            # I2 = np.multiply(I1, g2)
            if gpu_flag:
                g2 = cp.roll(g20, shift_x, axis=2)
                I2 = np.empty_like(I1)
                block_size = 5 
                for i in range(0, I1.shape[0], block_size):
                    start = i
                    end = min(i + block_size, I1.shape[0])
                    I1_block = cp.array(I1[start:end,:,:])
                    g2_block = cp.array(g2[start:end,:,:])
                    I2[start:end,:,:] = cp.asnumpy(cp.multiply(I1_block, g2_block))
            
            else: # The speed of numpy is not as fast as cupy, and it is not acceptable for large data
                g2 = np.roll(g20, shift_x, axis=2)
                I2 = np.empty_like(I1)
                block_size = 5  
                for i in range(0, I1.shape[0], block_size):
                    start = i
                    end = min(i + block_size, I1.shape[0])
                    I1_block = np.array(I1[start:end,:,:])
                    g2_block = np.array(g2[start:end,:,:])
                    I2[start:end,:,:] = np.multiply(I1_block, g2_block)

            PSC[ii,:,:] = detector(I2, ER, FOV, pixel_size, g2, dim) # shape: (nSteps, n_pixels, n_pixels)
            
            logger.info("Phase stepping: {}% ".format(round((ii+1)/len(np.arange(0, np.size(s)))*100), 2))
            print("\r", end="")
            print("Phase stepping: {}% ".format(round((ii+1)/len(np.arange(0, np.size(s)))*100), 2), end="")
            sys.stdout.flush()
        print()
        
        # Add detector noise after pixel integration and energy summation.
        if noise_flag == 1:
            if det_type == 'EnergyIntegral':
                if not np.isfinite(chi) or chi < 0:
                    raise ValueError('Gaussian noise std chi must be finite and nonnegative')
                PSC = PSC + np.random.normal(0.0, chi, size=PSC.shape)
            else:
                PSC = add_poission_noise(PSC, chi)
    
    return PSC
