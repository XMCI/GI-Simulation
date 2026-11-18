import numpy as np
import scipy.io
import cupy as cp
from functions.gpu_memory import gpu_memory
from scipy.interpolate import CubicSpline

def projected_source_kernel(n, dx, period, open_fraction, gap_values, fill_values):
    """Cell-integrated, centered G0 projection; its sum is the real throughput."""
    if not np.all(np.isfinite([dx, period, open_fraction])) or dx <= 0:
        raise ValueError('Source-kernel parameters must be finite and dx positive')
    if not 0 <= open_fraction <= 1:
        raise ValueError('The grating duty cycle must lie in [0, 1]')
    width = period / dx
    if width <= 0 or width + 1 >= n:
        raise ValueError('Projected G0 period must fit inside the simulation window')
    s = np.arange(n, dtype=float) - n // 2

    def overlap(w):
        return np.maximum(
            0.0, np.minimum(s + 0.5, w / 2) - np.maximum(s - 0.5, -w / 2)
        )

    whole = overlap(width)
    opening = overlap(width * open_fraction)
    gap = np.asarray(gap_values).reshape(-1, 1)
    bar = np.asarray(fill_values).reshape(-1, 1)
    return (gap * opening + bar * (whole - opening)) / width


def _real_intensity(conv):
    """Remove only FFT roundoff, never turn a negative intensity positive."""
    if not np.all(np.isfinite(conv)):
        raise ValueError('Source convolution produced nonfinite intensity')
    scale = max(float(np.max(np.abs(conv))), np.finfo(float).tiny)
    tolerance = 1e-10 * scale
    if np.max(np.abs(conv.imag)) > tolerance or np.min(conv.real) < -tolerance:
        raise ValueError('Source convolution must give nonnegative real intensity')
    return np.maximum(conv.real, 0.0)


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
    
    # G.dc is the bar fraction; retain the physical, energy-dependent throughput.
    if dim not in ('1D', '2D'):
        raise ValueError('dim must be 1D or 2D')
    dx = coord.x[0, 1] - coord.x[0, 0]
    if L <= 0 or D <= 0:
        raise ValueError('Source projection requires positive L and D')
    I1 = np.zeros_like(I0, dtype=np.float64)
    xp = cp if gpu_flag else np
    axis = I0.ndim - 1
    for e_start in range(0, nE, block_size):
        e_end = min(e_start + block_size, nE)
        kernel = projected_source_kernel(
            I0.shape[-1], dx, G.period * D / L, 1 - G.dc,
            gap_values[:, e_start:e_end], fill_values[:, e_start:e_end]
        )
        if dim == '2D':
            kernel = kernel[:, None, :]  # The same source kernel for every row.
        psf = xp.asarray(kernel)
        kernel_fft = xp.fft.fft(xp.fft.ifftshift(psf, axes=axis), axis=axis)
        # Retain row chunks to bound GPU memory without breaking the x convolution.
        n_rows = I0.shape[1] if dim == '2D' else 1
        for start in range(0, n_rows, 1024):
            end = min(start + 1024, n_rows)
            block = I0[e_start:e_end, start:end, :] if dim == '2D' else I0[e_start:e_end]
            block = xp.asarray(block)
            conv = xp.fft.ifft(xp.fft.fft(block, axis=axis) * kernel_fft, axis=axis)
            result = cp.asnumpy(conv) if gpu_flag else conv
            result = _real_intensity(result)
            if dim == '2D':
                I1[e_start:e_end, start:end, :] = result
            else:
                I1[e_start:e_end] = result
            del block, conv, result
        if gpu_flag:
            gpu_memory(f'Creating PSF {e_end/nE*100:.1f}%', logger)
        else:
            logger.info(f'Creating PSF with CPU: {e_end/nE*100:.1f}%')
        del psf, kernel_fft
    if gpu_flag:
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
    return I1
