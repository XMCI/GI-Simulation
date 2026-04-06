import numpy as np
import cupy as cp
import pycuda.driver as drv
import scipy.io
from scipy.interpolate import CubicSpline
from functions.gpu_memory import gpu_memory
from tqdm import tqdm

def create_grating(G, E, coord, dim, gpu_flag, logger, abs_flag=0, block_size=5): 
    '''
        DESCRIPTION:
            generates grating transmission function. The oblique incidence condition of X-rays on the grating was not considered.

        CALL: 
            - G: grating parameter struct
                - G.period: pitch of grating
                - G.dc: duty cycle of grating
                - G.material: plated material in the grating surface
                - G.thickness: the thickness of plated material
                - G.base_material: base material of the grating, default 'air'
                - G.base_thickness: the thickness of the base material, default 0
            - E: vector containing energy range
            - coord: spatial range for which grating is designed
            - gpu_flag: logical variable whether using the GPU or not
            - logger: logger for logging information
            - abs_flag: logical variable whether to return the absolute value of the grating transmission function, default 0
            - block_size: size of the block for 2D processing the grating, default 5, which deoends on the GPU memory size.
        OUTPUT:
            - gt: grating as a transmission function (complex array) with shape (nE, len(x)) for 1D grating, or (nE, Nx, Ny) for 2D grating

        UPDATES:
            2025/06/01 (Longchao Men): optimize the GPU memory usage
            2024/11/27 (Longchao Men): add the 2D grating generation
            2021/11/12 (Chengpeng Wu): add the gpu_flag and corresponding operations
            2021/10/29 (Chengpeng Wu): first version
    '''

    print('Create Grating...')
    # constant
    if isinstance(E, np.ndarray):  
        nE = E.shape[1]
        lambda_ = (1.239842 / E) * 1e-9  # shape (1, nE)
    else:
        nE = 1
        lambda_ = np.array((1.239842 / E) * 1e-9).reshape(1,1)  # shape (1,1)
    k = np.sin((np.pi - np.multiply(2*np.pi, G.dc)) / 2) 

    if not G.base_material:
        G.base_material = 'Air'
        G.base_thickness = 0
    if G.base_thickness > 0:
        base_mat_file = './/XOPDATA//'+G.base_material+'Data.mat'
        MatData_file = scipy.io.loadmat(base_mat_file)
        MatData = MatData_file['MatData'] 
        sc = CubicSpline(MatData[:,0], MatData[:,1])
        base_delta = sc(E*1000)
        sc = CubicSpline( MatData[:,0], MatData[:,3])
        base_beta = np.multiply(sc(E*1000), 100) * lambda_ / (4*np.pi)
    else:
        base_delta = 0
        base_beta = 0
    
    mat_file = './/XOPDATA//' + G.material + 'Data.mat'
    MatData_file = scipy.io.loadmat(mat_file)
    MatData = MatData_file['MatData'] 
    sc = CubicSpline(MatData[:,0], MatData[:,1])
    delta = sc(E*1000)
    sc = CubicSpline(MatData[:,0], MatData[:,3])
    beta = np.multiply(sc(E*1000), 100) * lambda_ / (4*np.pi)
    # calculate the fill and gap values
    fill_values = np.exp(np.multiply(np.multiply(1j,2.0) * np.pi / lambda_,(np.multiply((-delta + np.multiply(1j,beta)),G.thickness) + np.multiply((- base_delta + np.multiply(1j,base_beta)), G.base_thickness)))).reshape(1,-1) # shape: (1,nE)
    gap_values = np.exp(np.multiply(np.multiply(np.multiply(1j,2.0) * np.pi / lambda_,(-base_delta + np.multiply(1j,base_beta))), G.base_thickness)).reshape(1,-1) # shape: (1,nE)

    # generate 0-1 mask
    if dim == '1D':
        mask = (np.sin(np.multiply(2*np.pi, coord.x) / G.period) > k)
        mask = np.tile(mask, [np.size(E), 1])

        gt = np.zeros(np.shape(mask), dtype = np.complex128)
        fill_values = np.tile(np.transpose(fill_values), [1, np.size(coord.x, axis=1)]) # shape: (nE,nP)
        gap_values = np.tile(np.transpose(gap_values), [1, np.size(coord.x, axis=1)]) # shape: (nE,nP)

        if gpu_flag:
            mask = cp.asarray(mask)
            gt = cp.asarray(gt)
            fill_values = cp.asarray(fill_values)
            gap_values = cp.asarray(gap_values)
            gpu_memory('Create Grating 1D', logger)

            gt[mask == 1] = fill_values[mask == 1]
            gt[mask == 0] = gap_values[mask == 0]

            del mask, fill_values, gap_values
            if abs_flag:
                gt = cp.abs(gt)**2
            gt = cp.asnumpy(gt)
        
        else:
            logger.info('Create Grating with CPU...')
            # the following computation is time-consuming and can be much faseter using the GPU arrays
            gt[mask == 1] = fill_values[mask == 1]
            gt[mask == 0] = gap_values[mask == 0]
            if abs_flag:
                gt = np.abs(gt)**2

    if dim == '2D': 
        Nx, Ny = coord.x.shape[1], coord.y.shape[0]
        if isinstance(E, np.ndarray):  
            nE = E.shape[1]
        else:
            nE = 1
        gt = np.zeros((nE, Nx, Ny), dtype=np.complex64)

        x_grid = np.tile(coord.x[0, :][None, :], (Ny, 1))  # shape: (Nx, Ny)
        mask2d = (np.sin(2 * np.pi * x_grid / G.period) > k)  # shape: (Nx, Ny)

        for e0 in range(0, nE, block_size):
            if gpu_flag:
                e1 = min(nE, e0 + block_size)
                mask_gpu = cp.asarray(mask2d[None, :, :])         # shape: (1, Nx, Ny)
                mask_gpu = cp.tile(mask_gpu, (e1 - e0, 1, 1))     # shape: (B, Nx, Ny)

                fill_vals = cp.asarray(fill_values[0, e0:e1])[:, None, None]  # shape: (B,1,1)
                gap_vals = cp.asarray(gap_values[0, e0:e1])[:, None, None]

                fill_block = cp.tile(fill_vals, (1, Nx, Ny))      # shape: (B, Nx, Ny)
                gap_block = cp.tile(gap_vals, (1, Nx, Ny))        # shape: (B, Nx, Ny)
                
                gt_block = cp.where(mask_gpu, fill_block, gap_block)
                
                if abs_flag:
                    gt = np.zeros((nE, Nx, Ny), dtype=np.float32)
                    chunk_size = 1024  
                    total_cols = gt_block.shape[2]
                    for start in range(0, total_cols, chunk_size):
                        end = min(start + chunk_size, total_cols)
                        gt_chunk = gt_block[:, :, start:end]
                        # Calculate the absolute value squared for the chunk
                        gt_chunk = cp.abs(gt_chunk) ** 2
                        gt_block[:, :, start:end] = gt_chunk.real.astype(np.float32)
                    # gt_block = cp.abs(gt_block)**2
                gt[e0:e1] = cp.asnumpy(gt_block)

                gpu_memory(f'Create 2D Grating processing {e1/nE*100:.1f}%', logger)
                del mask_gpu, fill_block, gap_block, gt_block
                cp.get_default_memory_pool().free_all_blocks()
            
            else:
                e1 = min(nE, e0 + block_size)
                mask_cpu = np.asarray(mask2d[None, :, :])         # shape: (1, Nx, Ny)
                mask_cpu = np.tile(mask_cpu, (e1 - e0, 1, 1))     # shape: (B, Nx, Ny)

                fill_vals = np.asarray(fill_values[0, e0:e1])[:, None, None]  # shape: (B,1,1)
                gap_vals = np.asarray(gap_values[0, e0:e1])[:, None, None]

                fill_block = np.tile(fill_vals, (1, Nx, Ny))      # shape: (B, Nx, Ny)
                gap_block = np.tile(gap_vals, (1, Nx, Ny))        # shape: (B, Nx, Ny)
                
                gt_block = np.where(mask_cpu, fill_block, gap_block)
                if abs_flag:
                    gt = np.zeros((nE, Nx, Ny), dtype=np.float32)
                    gt_block = np.abs(gt_block)**2
                gt[e0:e1] = gt_block

                del mask_cpu, fill_block, gap_block, gt_block
                logger.info(f'Create 2D Grating with CPU: {e1/nE*100:.1f}%')

    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    return gt