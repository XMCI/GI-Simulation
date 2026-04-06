import numpy as np
import sys
# from functions.create_grating import create_grating
from functions.create_grating_v2 import create_grating
from functions.detector import detector
from functions.add_poission_noise import add_poission_noise
import cupy as cp
    
def phase_stepping(I1, nSteps, nPeriods, E, coord, G2, ER, pixel_size, chi, dim, gpu_flag, noise_flag, logger): 
    '''
        DESCRIPTION: 
            this function performs the phase stepping and the detection of the signal. 
            Poisson noise is also added at this step.
        
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
            - chi: proportionality factor depending on detector properties
            - gpu_flag: logical variable whether using the GPU or not
            - noise_flag: add poisson noise or not, if 1 or no input, then add noise
        OUTPUT:
            - PSC: phase stepping curve
            
        UPDATES:
        2024/11/27 (Longchao Men): add the 2D mode
        2021/11/12 (Chengpeng Wu): add the gpu_flag and corresponding operations
        2021/11/09 (Chengpeng Wu): create G2 and I1 before phase-stepping to avoid repeated calculation
        2021/11/01 (Chengpeng Wu): first version
    '''
    
    if not 'noise_flag' in locals():
        noise_flag = 1
    
    if dim == '1D':
        FOV = np.amax(coord.x)
        dx = coord.x[0,1]-coord.x[0,0]
        # vector containing the phase step distances
        s = np.linspace(0, G2.period*nPeriods, nSteps*nPeriods+1)
        s = s[:-1]

        PSC = np.zeros((len(np.arange(0, FOV, pixel_size)), np.size(s)))
        g20 = create_grating(G2, E, coord, dim, gpu_flag, logger, abs_flag=1)

        for ii in np.arange(0, np.size(s)):
            shift_x = np.round(s[ii] / dx).astype(int)
            g2 = np.roll(g20, shift_x, axis=1)
            PSC[:,ii] = detector(I1, ER, FOV, pixel_size, g2, dim) # shape: (n_pixels, nSteps)

            print("\r", end="")
            print("Phase stepping: {}% ".format(round((ii+1)/len(np.arange(0, np.size(s)))*100), 2), end="")
            sys.stdout.flush()
        print()

        # add poission noise
        if noise_flag == 1:
            PSC = add_poission_noise(PSC, chi)
        
    elif dim == '2D':
        FOV = np.amax(coord.x)
        dx = coord.x[0,1]-coord.x[0,0]
        dy = coord.y[1,0]-coord.y[0,0]
        # vector containing the phase step distances
        s = np.linspace(0, G2.period*nPeriods, nSteps*nPeriods+1)
        s = s[:-1]

        n_pixels = np.round(FOV/pixel_size).astype(int)
        PSC = np.zeros((np.size(s), n_pixels, n_pixels))
        g20 = create_grating(G2, E, coord, dim, gpu_flag, logger, abs_flag=1) #abs_flag is used to create the absolute value of the grating
        if gpu_flag:
            g20 = cp.asarray(g20)

        for ii in np.arange(0, np.size(s)):
            shift_x = np.round(s[ii] / dy).astype(int)

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
        
        # add poission noise
        if noise_flag == 1:
            PSC = add_poission_noise(PSC, chi)
    
    return PSC

    
    