import numpy as np
    
def detector(I2, ER, FOV, pixel_size, g2, dim): 
    '''
        DESCRIPTION:
            simulates per pixel in detector plane.

        CALL : 
            - I1: incident photon intensity
            - ER: detector response
            - FOV: field of view
            - pixel_size: detector pixel size
            - psf: vector containing point spread function of the source after G0
            - G2: the G2 grating struct
                - G2.period: pitch of grating
                - G2.dc: duty cycle of grating
                - G2.material: plated material in the grating surface
                - G2.thickness: the thickness of plated material
                - G2.base_material: base material of the grating, default 'air'
                - G2.base_thickness: the thickness of the base material, default 0
            - PSC: phase stepping curve

        UPDATES:
            2024/11/27 (Longchao Men): add the 2D mode
            2021/11/01 (Chengpeng Wu): first version
    '''
    nPixel = np.round(FOV/pixel_size).astype(int)
    ER = np.tile(np.transpose(ER), (1, np.size(I2, 1)))
    # I2 = np.multiply(I1, g2)
    
    if dim == '1D':
        # take into consideration the spectrum and detector response
        I2 = np.squeeze(np.sum(np.multiply(ER, I2), axis=0)).reshape(1,-1) # shape: (1, nP)
        # pixel indiexs
        pxl_points = np.divide(np.size(I2, 1), nPixel).astype(int) # the number of points in a pixel
        pxl_inds = np.arange(0, nPixel)
        lower_pxl = np.floor(pxl_inds * pxl_points).astype(int)
        upper_pxl = np.ceil((pxl_inds+1) * pxl_points-1).astype(int)

        # handle the condition outside the bound
        if upper_pxl[-1] < np.size(I2, 1):
            I2 = np.delete(I2, np.arange(upper_pxl[-1]+1, np.size(I2))).reshape(1,-1)
        else:
            start_index = np.size(I2)  
            end_index = upper_pxl[0] - lower_pxl[1] + (upper_pxl[-1] - lower_pxl[-1])
            I2[start_index:end_index] = 0
        
        I2 = np. reshape(I2, (pxl_points, nPixel), order='F')
        # sum values of all points in a pixel
        PSC = np.sum(I2, 0)

    elif dim == '2D':
        # take into consideration the spectrum and detector response
        ER = np.tile(np.expand_dims(ER, axis=2), (1, 1, np.size(I2, 2)))

        I2 = np.squeeze(np.sum(np.multiply(ER, I2), axis=0)) # shape: (nP, nP)

        #pixel indices
        pxl_x = np.divide(np.size(I2, 0), nPixel).astype(int) # the number of points in a pixel in x direction
        pxl_y = np.divide(np.size(I2, 1), nPixel).astype(int) # the number of points in a pixel in y direction
        ipx_x, ipx_y = np.arange(0, nPixel), np.arange(0, nPixel)

        start_idx_x = np.floor(ipx_x * pxl_x).astype(int)
        end_idx_x = np.ceil(pxl_x * (ipx_x + 1) - 1).astype(int)
        start_idx_y = np.floor(ipx_y * pxl_y).astype(int)
        end_idx_y = np.ceil(pxl_y * (ipx_y + 1) - 1).astype(int)

        # Handle boundary conditions for x dimension
        if end_idx_x[-1] < np.size(I2, 0):
            I2 = np.delete(I2, np.arange(end_idx_x[-1]+1, I2.shape[0]), axis=0)
        else:
            extra_rows = end_idx_x[-1] - I2.shape[0]
            I2 = np.pad(I2, ((0, extra_rows), (0, 0)), mode='constant', constant_values=0)
        # Handle boundary conditions for y dimension
        if end_idx_y[-1] < np.size(I2, 1):
            I2 = np.delete(I2, np.arange(end_idx_y[-1]+1, I2.shape[1]), axis=1)
        else:
            extra_cols = end_idx_y[-1] - I2.shape[1]
            I2 = np.pad(I2, ((0, 0), (0, extra_cols)), mode='constant', constant_values=0)

        # # Calculate signal at each pixel
        I2 = np. reshape(I2, (nPixel, pxl_x, nPixel, pxl_y))
        # # sum values of all points in a pixel
        PSC = np.sum(I2, axis=(1, 3))
    
    return PSC
    