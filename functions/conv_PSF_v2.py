import numpy as np
import scipy.io
import cupy as cp
from functions.gpu_memory import gpu_memory
from scipy.interpolate import CubicSpline

def conv_PSF(I0, G, E, coord, L, D, dim, gpu_flag, logger, block_size = 5): 
    '''
        DESCRIPTION:
            generates projected source intensity distribution for convolution, only consider one period of G0 as the fringes of other periods are the same.

        CALL: 
            - G: grating G0 parameter struct
                - G.period: pitch of grating
                - G.dc: duty cycle of grating
                - G.material: plated material in the grating surface
                - G.thickness: the thickness of plated material
                - G.base_material: base material of the grating, default 'air'
                - G.base_thickness: the thickness of the base material, default 0
            - E: vector containing energy range
            - x: spatial range for which grating is designed
            - L: distance between G0 and G1
            - D: distance between G1 and G2
            - gpu_flag: logical variable whether using the GPU or not
        OUTPUT:
            - psf: projected source distribution

        UPDATES:
            2025/06/01 (Longchao Men): optimize the GPU memory usage
            2024/11/27 (Longchao Men): add the 2D mode
            2021/11/15 (Chengpeng Wu): fixed the intensity problem by multiplying G.dc
            2021/11/12 (Chengpeng Wu): add the gpu_flag and corresponding operations
            21/10/29 (Chengpeng Wu): first version

    '''

    print('Creating PSF and convoluting with I0...')
    # constant
    if isinstance(E, np.ndarray):  
        nE = E.shape[1]
        lambda_ = (1.239842 / E) * 1e-9  # shape (1, nE)
    else:
        nE = 1
        lambda_ = np.array((1.239842 / E) * 1e-9).reshape(1,1)  # shape (1,1)

    
    if not G.base_material:
        G.base_material = 'Air'
        G.base_thickness = 0
    if G.base_thickness > 0:
        base_mat_file = 'XOPDATA/'+G.base_material+'Data.mat'
        MatData_file = scipy.io.loadmat(base_mat_file)
        MatData = MatData_file['MatData']

        sc = CubicSpline(MatData[:,0], MatData[:,1])
        base_delta = sc(E*1000)
        sc = CubicSpline( MatData[:,0], MatData[:,3])
        base_beta = np.multiply(sc(E*1000), 100) * lambda_ / (4*np.pi)
    else:
        base_delta = 0
        base_beta = 0
    
    mat_file = 'XOPDATA/'+G.material+'Data.mat'
    MatData_file = scipy.io.loadmat(mat_file)
    MatData = MatData_file['MatData']

    sc = CubicSpline(MatData[:,0], MatData[:,1])
    delta = sc(E*1000)
    sc = CubicSpline(MatData[:,0], MatData[:,3])
    beta = np.multiply(sc(E*1000), 100) * lambda_ / (4*np.pi)

    fill_values = np.exp(np.multiply(-4*np.pi / lambda_, (np.multiply(base_beta, G.base_thickness) + np.multiply(beta,G.thickness)))).reshape(1,-1)  # shape: (1,nE)
    gap_values = np.exp(np.multiply(np.multiply(-4*np.pi / lambda_, base_beta), G.base_thickness)).reshape(1,-1) # shape: (1,nE)
    
    # generate 0-1-2 mask
    if dim == '1D':
        dx = coord.x[0, 1] - coord.x[0, 0]
        mask = np.zeros(np.shape(coord.x[0, :])).reshape(1, -1)
        nP = np.size(coord.x, axis=1)
        proj_period = G.period / dx * D / L
        proj_gap = G.period * G.dc / dx * D / L

        mask[0, np.arange(np.round(nP / 2 - proj_period / 2) - 1, np.round(nP / 2 + proj_period / 2 - 1)).astype(int)] = 2
        mask[0, np.arange(np.round(nP / 2 - proj_gap / 2) - 1, np.round(nP / 2 + proj_gap / 2 - 1)).astype(int)] = 1
        mask = np.tile(mask, (np.size(E), 1))

        psf = np.zeros(np.shape(mask))
        fill_values = np.tile(np.transpose(fill_values), [1, np.size(coord.x, axis=1)])
        gap_values = np.tile(np.transpose(gap_values), [1, np.size(coord.x, axis=1)])

        if gpu_flag:
            mask = cp.asarray(mask)
            psf = cp.asarray(psf)
            fill_values = cp.asarray(fill_values)
            gap_values = cp.asarray(gap_values)
            gpu_memory('Creating PSF and convoluting with I0...', logger)

            psf[mask == 1] = gap_values[mask == 1]
            psf[mask == 2] = fill_values[mask == 2]

            del mask, fill_values, gap_values
            psf = psf / cp.sum(psf, axis=1).reshape(-1, 1) * G.dc
            I1 = cp.abs(cp.fft.fftshift(cp.fft.ifft(cp.fft.fft(cp.asarray(I0), axis=1) * cp.fft.fft(psf, axis=1), axis=1), axes=1))
            return cp.asnumpy(I1)

        else:
            logger.info('Creating PSF and convoluting with I0 using CPU...')
            psf = psf / np.sum(psf, axis=1).reshape(-1, 1) * G.dc
            I1 = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.fft(I0, axis=1) * np.fft.fft(psf, axis=1), axis=1), axes=1))

    elif dim == '2D':
        if isinstance(E, np.ndarray):  
            nE = E.shape[1]
        else:
            nE = 1
        dx = coord.x[0, 1] - coord.x[0, 0]
        dy = coord.y[0, 1] - coord.y[0, 0]
        nx = coord.x.shape[1]
        ny = coord.y.shape[1]

        proj_period = G.period / dx * D / L
        proj_gap = G.period * G.dc / dx * D / L
        e_blocks = int(np.ceil(nE / block_size))

        I1 = np.zeros_like(I0, dtype=np.float64)

        for e in range(e_blocks):
            e_start = e * block_size
            e_end = min((e + 1) * block_size, nE)

            fill_e = fill_values[:, e_start:e_end]
            gap_e = gap_values[:, e_start:e_end]

            try:
                mask = np.zeros((nx, ny))
                x_center = nx // 2
                gap_range = np.arange(np.round(x_center - proj_gap / 2) - 1,
                                    np.round(x_center + proj_gap / 2) - 1).astype(int)
                period_range = np.arange(np.round(x_center - proj_period / 2) - 1,
                                        np.round(x_center + proj_period / 2) - 1).astype(int)
                mask[:, period_range] = 2
                mask[:, gap_range] = 1

                mask = np.tile(mask[np.newaxis, :, :], (e_end - e_start, 1, 1))
                psf = np.zeros_like(mask, dtype=np.float64)

                fill_tile = np.tile(fill_e.T[:, :, np.newaxis], [1, nx, ny])
                gap_tile = np.tile(gap_e.T[:, :, np.newaxis], [1, nx, ny])

                # create G0 (PSF)
                if gpu_flag:
                    mask = cp.asarray(mask)
                    fill_tile = cp.asarray(fill_tile)
                    gap_tile = cp.asarray(gap_tile)
                    psf = cp.asarray(psf)

                psf[mask == 1] = gap_tile[mask == 1]
                psf[mask == 2] = fill_tile[mask == 2]

                # convoluting with I0
                if gpu_flag:
                    del mask, fill_tile, gap_tile
                    psf /= cp.expand_dims(cp.sum(psf, axis=2), axis=2)
                    psf *= G.dc

                    I0_block = cp.asarray(I0[e_start:e_end, :, :])
                    
                    # Implement convolution for each layer
                    chunk_size = 1024
                    conv_block = cp.zeros_like(I0_block, dtype = np.complex128)
                    for start in range(0, I0_block.shape[2], chunk_size):
                        end = min(start + chunk_size, I0_block.shape[2])

                        I0_chunk = I0_block[:, start:end, :]
                        psf_chunk = psf[:, start:end, :]

                        I0_fft = cp.fft.fft(I0_chunk, axis=2)
                        psf_fft = cp.fft.fft(psf_chunk, axis=2)
                        conv_chunk = cp.fft.ifft(I0_fft * psf_fft, axis=2)
                        conv_block[:, start:end, :] = conv_chunk
                    # conv_block = cp.fft.ifft(cp.fft.fft(I0_block, axis=2) * cp.fft.fft(psf, axis=2), axis=2)
                    del psf, I0_block

                    conv_block = cp.fft.fftshift(conv_block, axes=2)
                    conv_result = cp.abs(conv_block).get()

                    I1[e_start:e_end, :, :] = cp.asnumpy(conv_result)

                    gpu_memory(f'Creating PSF and convoluting with I0 processing {e_end/nE*100:.1f}%', logger)
                    del conv_block, conv_result
                    cp.cuda.Stream.null.synchronize()
                    cp.get_default_memory_pool().free_all_blocks()
                
                else:
                    psf = psf / (np.sum(psf, axis=2, keepdims=True))
                    psf *= G.dc
                    I0_block = I0[e_start:e_end, :, :]
                    conv_block = np.fft.ifft(np.fft.fft(I0_block, axis=2) * np.fft.fft(psf, axis=2), axis=2)
                    conv_block = np.fft.fftshift(conv_block, axes=2)
                    I1[e_start:e_end, :, :] = np.abs(conv_block)
                    logger.info(f'Creating PSF and convoluting with I0 processing {e_end/nE*100:.1f}%')

            except Exception as err:
                logger.error(f"Error in energy block {e+1}/{e_blocks}: {str(err)}")
                raise
    
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    return I1
