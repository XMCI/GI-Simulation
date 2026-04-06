import numpy as np
import cupy as cp
from functions.gpu_memory import gpu_memory
    
def fresnel_propagation_poly(Wi, E, FOV, L, D, srcType, dim, gpu_flag, logger, batch_size = 5): 
    '''
        DESCRIPTION: 
            polychromatic propagation for 1D waves, considering both thepoint source and 
            plane wave situations.

        CALL: 
            - Wi: input wavefront
                - 1D: shape (nE, nP)
                - 2D: shape (nE, nP, nP)
            - E: vector containing energy range
            - FOV: field of view
            - L: distance between G0 and G1
            - D: distance between G1 and G2
            - type: 'pointsource' or 'planewave'
            - gpu_flag: logical variable whether using the GPU or not
            - dim: dimension of the wavefront, 1 or 2
        OUTPUT:
            - Wo: output wavefront
                - 1D: shape (nE, nP)
                - 2D: shape (nE, nP, nP)

        UPDATES:
            2025/06/01 (Longchao Men): optimize the GPU memory usage
            2024/10/28 (Longcho Men): add the 2D wavefront propagation
            2022/07/04 (Peiyuan Guo): add the z item at the fresnel propagator
            2021/11/12 (Chengpeng Wu): add the gpu_flag and corresponding operations
            2021/11/01 (Chengpeng Wu): first version
    '''
    if isinstance(E, np.ndarray):  
        nE = E.shape[1]
        lambda_ = (1.239842 / E) * 1e-9  # shape (1, nE)
    else:
        nE = 1
        lambda_ = np.array((1.239842 / E) * 1e-9).reshape(1,1)  # shape (1,1)
    print('Fresnel Propagation...')

    if dim == '1D':
        nP = np.size(Wi, axis=1)
        dx = FOV / nP
        fx = np.linspace(-1/2/dx, 1/2/dx, nP).reshape(1,-1)
        fx = np.tile(fx, (np.size(lambda_), 1))
        lambda_ = np.tile(np.transpose(lambda_), (1, nP))

        if srcType == 'planewave':
            z = D
            if gpu_flag:
                z = cp.asarray(z)
                lambda_ = cp.asarray(lambda_)
                fx = cp.asarray(fx)

                H = cp.exp(cp.multiply(cp.multiply(cp.multiply(-1j*cp.pi, lambda_), z), fx**2) + 1j*2*cp.pi/lambda_*z)
                H = cp.fft.fftshift(H, axes=1)
                FWi = cp.fft.fft(cp.asarray(Wi), axis=1)
                FWo = cp.multiply(FWi, H)
                Wo = cp.fft.ifft(FWo, axis=1)

                gpu_memory('Fresnel Propagation 1D', logger)
                del H, FWi, FWo
                Wo = cp.asnumpy(Wo)
            
            else:
                logger.info('Fresnel Propagation...')

                H = np.exp(np.multiply(np.multiply(np.multiply(-1j*np.pi, lambda_), z), fx**2) + 1j*2*np.pi/lambda_*z)
                H = np.fft.fftshift(H, axes=1)
                Fwi = np.fft.fft(Wi, axis=1)
                Fwo = np.multiply(Fwi, H)
                Wo = np.fft.ifft(Fwo, axis=1)

    elif dim == '2D':
        nP_x = Wi.shape[1]
        nP_y = Wi.shape[2]
        dx = FOV / nP_x
        dy = FOV / nP_y

        fx = np.linspace(-0.5 / dx, 0.5 / dx, nP_x)
        fy = np.linspace(-0.5 / dy, 0.5 / dy, nP_y)
        FX, FY = np.meshgrid(fx, fy, indexing='ij')  # shape (nP_x, nP_y)

        Wo = np.zeros_like(Wi, dtype=np.complex128)

        for i in range(0, nE, batch_size):
            i_end = min(i + batch_size, nE)
            lambda_batch = lambda_[0, i:i_end].reshape(-1, 1, 1)  # shape (b,1,1)

            if gpu_flag:
                lb = cp.asarray(lambda_batch)
                FX_cp = cp.asarray(FX)
                FY_cp = cp.asarray(FY)

                H = cp.exp(-1j * cp.pi * lb * (FX_cp**2 + FY_cp**2) * D + 1j * 2 * cp.pi / lb * D)
                del lb, FX_cp, FY_cp
                H = cp.fft.fftshift(H, axes=(1, 2))

                Wi_cp = cp.asarray(Wi[i:i_end, :, :])
                FWi = cp.fft.fftn(Wi_cp, axes=(1, 2))
                del Wi_cp

                FWo = FWi * H
                del FWi, H

                Wo_batch = cp.fft.ifftn(FWo, axes=(1, 2))
                del FWo
                Wo[i:i_end, :, :] = cp.asnumpy(Wo_batch)

                gpu_memory(f'Fresnel Propagation 2D processing {i_end/nE*100:.1f}%', logger)
                del Wo_batch
                cp._default_memory_pool.free_all_blocks()
            
            else:
                logger.info(f'Fresnel Propagation 2D with CPU: {i_end/nE*100:.1f}%')
                H = np.exp(-1j * np.pi * lambda_batch * (FX**2 + FY**2) * D + 1j * 2 * np.pi / lambda_batch * D)
                H = np.fft.fftshift(H, axes=(1, 2))
                FWi = np.fft.fftn(Wi[i:i_end, :, :], axes=(1, 2))
                FWo = FWi * H
                Wo[i:i_end, :, :] = np.fft.ifftn(FWo, axes=(1, 2))
    
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    return Wo
    