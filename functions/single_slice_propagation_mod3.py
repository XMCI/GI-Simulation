import numpy as np
import cupy as cp
import scipy.io
from scipy.interpolate import CubicSpline, interp1d
from scipy.interpolate import RegularGridInterpolator as RGI
from functions.gpu_memory import gpu_memory
    
def single_slice_propagation_mod3(Wi, E, FOV, z, dz, type_, phant, propa_type, dim, gpu_flag, logger): 
    '''
        DESCRIPTION: 
        SINGLE_SLICE_PROPAGATION_MOD2  wave optics simulation, a single slice propagation which is used in multi-slice approximation.
        In Murno's article, the thickness of the phantom should not be demagnified, different from the propagation distance.

        CALL:
        - Wi: input wavefront, DIM 2 spatial dimension, DIM 1 energy dimension
        - E: vector containing energy range
        - FOV: field of view, unit: m
        # - dz: thickness of this slice, unit: m
            (it can be calculated by other variables)
        - z0: the distance the ray have propagated (useless)
        - dz: the propagation of this slice, only useful when point source mode (but the type is actually useless). In plane wave mode can input [] instead.
        - type: 'pointsource' or 'planewave'
        - gpu_flag: logical variable whether using the GPU or not
        - phant: a struct, containing a slice of the phantom and the material of the phantom and the voxel size of the phantom. The phantom matrix is a binary matrix. The phantom should be located between G1 and G2.
        - propa_type: propagation type, 'frensel' or 'projection'
        OUTPUT:
        - Wo: output wavefront

        Updates:
        2024/10/28 (Longchao Men): add the 2D mode
        2023/05/04 (Peiyuan Guo): add gpu method
    '''
    if not isinstance(E, np.ndarray):
        E = np.array([E]).reshape(1,-1)
    lambda_ = np.multiply(np.divide(1.239842, E), 1e-09).reshape(1,-1)

    ## load refractive index corresponding to the energy and fill the phantom with the refractive index
    mat_file = './/XOPDATA//' + phant.material + 'Data.mat'
    MatData_file = scipy.io.loadmat(mat_file)
    MatData = MatData_file['MatData']
    sc = CubicSpline(MatData[:,0], MatData[:,1])
    delta = np.reshape(sc(E*1000), (1,-1)) # shape: (1,len(E))
    sc = CubicSpline(MatData[:,0], MatData[:,3])
    beta = np.reshape(np.multiply(sc(E*1000), 100), (1,-1))* lambda_/(4*np.pi) # shape: (1,len(E))
    # delta - i beta
    complex_refractive_index_term = delta - 1j * beta

    if dim == '1D':
        nP = np.size(Wi, axis=1)
        dx = FOV / nP
        fx = np.linspace(-1/2/dx, 1/2/dx, nP).reshape(1,-1)
        fx = np.tile(fx, (np.size(lambda_), 1))
        lambda_ = np.tile(np.transpose(lambda_), (1, nP))
    
        ## operate on the phantom
        # integrate the phantom along it's thickness
        phant_int = np.sum(phant.phantom, axis=0) * phant.dx
        # interpolate the phantom with the gird dx
        phant_int = phant_int.reshape(1,-1)
        nP_phantom = np.size(phant_int, 1)
        x_old = np.squeeze(np.linspace(-nP_phantom/2, nP_phantom/2-1, nP_phantom)*phant.dx)
        x_new = np.squeeze(np.linspace(-FOV/2, FOV/2-dx, nP))
        phant_used = np.zeros_like(x_new)
        phant_interp = interp1d(x_old, np.squeeze(phant_int), kind='nearest', fill_value='extrapolate')
        # the mask is used to select the phantom range in the new grid
        mask = np.squeeze((x_new >= -nP_phantom/2*phant.dx) & (x_new <= (nP_phantom/2 - 1)*phant.dx))
        # interpolate the with phantom range 
        phant_used[mask] = phant_interp(x_new[mask])
        del x_old, x_new, phant_interp, mask
        phant_used, complex_refractive_index_term = np.meshgrid(phant_used, complex_refractive_index_term)
        phant_used = np.multiply(phant_used, complex_refractive_index_term)
        
        ## fresnel propogate
        if dz == 0:
            dz = np.size(phant.phantom, 0) * phant.dx
        
        if propa_type == 'fresnel':
            if gpu_flag:
                dz = cp.asarray(dz)
                lambda_ = cp.asarray(lambda_)
                fx = cp.asarray(fx)
                phant_used = cp.asarray(phant_used)
                Wi = cp.asarray(Wi)
                gpu_memory('multi_slice Propagation',logger)

                H = cp.exp(cp.multiply(cp.multiply(cp.multiply(-1j*cp.pi, lambda_), dz), fx**2) + 1j*2*cp.pi/lambda_*dz)
                H = cp.fft.fftshift(H, axes=1)
                Wi = cp.multiply(Wi, cp.exp(cp.multiply(-1j*2*cp.pi/lambda_, phant_used)))
                FWi = cp.fft.fft(Wi, axis=1)
                FWo = cp.multiply(FWi, H)
                Wo = cp.asnumpy(cp.fft.ifft(FWo, axis=1))
                del H, FWi, FWo
            else:
                H = np.exp(np.multiply(np.multiply(np.multiply(-1j*np.pi, lambda_), dz), fx**2) + 1j*2*np.pi/lambda_*dz)
                H = np.fft.fftshift(H, axes=1)
                Wi = np.multiply(Wi, np.exp(np.multiply(- 1j*2*np.pi/lambda_, phant_used)))
                Fwi = np.fft.fft(Wi, axis=1)
                Fwo = np.multiply(Fwi, H)
                Wo = np.fft.ifft(Fwo, axis=1)
        ### not using this part
        else:
            if propa_type == 'projection' or propa_type == 'quasi-projection':
                Wi = np.multiply(Wi,np.exp(np.multiply(- 1j * 2 * np.pi / lambda_,phant_used)))
                Wo = np.multiply(Wi,np.exp(1j * 2 * np.pi / lambda_ * dz))
    
    elif dim == '2D':
        nP_x = np.size(Wi, axis=1)
        nP_y = np.size(Wi, axis=2)
        dx = FOV / nP_x
        dy = FOV / nP_y
        fx = np.linspace(-1/2/dx, 1/2/dx, nP_x).reshape(1,-1)
        fy = np.linspace(-1/2/dy, 1/2/dy, nP_y).reshape(1,-1)
        fx, fy = np.meshgrid(fx, fy)
        fx = np.tile(fx, (np.size(lambda_), 1, 1))
        fy = np.tile(fy, (np.size(lambda_), 1, 1))
        lambda_ = np.tile(np.expand_dims(np.transpose(lambda_), axis=2), (1, nP_x, nP_y))
        
        ## operate on the phantom
        # integrate the phantom along it's thickness
        phant_int = np.sum(phant.phantom, axis=0) * phant.dx
        # interpolate the phantom with the gird dx
        nPx_phantom = np.size(phant_int, 0)
        nPy_phantom = np.size(phant_int, 1)
        x_old = np.squeeze(np.linspace(-nPx_phantom/2, nPx_phantom/2-1, nPx_phantom)*phant.dx)
        y_old = np.squeeze(np.linspace(-nPy_phantom/2, nPy_phantom/2-1, nPy_phantom)*phant.dx)
        phant_interp = RGI((x_old, y_old), phant_int, method='nearest', bounds_error=False)
        x_new = np.squeeze(np.linspace(-FOV/2, FOV/2-dx, nP_x))
        y_new = np.squeeze(np.linspace(-FOV/2, FOV/2-dy, nP_y))
        phant_used = np.zeros((len(x_new), len(y_new)))
        x_new, y_new = np.meshgrid(x_new, y_new)
        # select the values in the range of the phantom
        maskx = (x_new >= -nPx_phantom/2*phant.dx) & (x_new <= (nPx_phantom/2 - 1)*phant.dx)
        masky = (y_new >= -nPy_phantom/2*phant.dx) & (y_new <= (nPy_phantom/2 - 1)*phant.dx)
        mask = maskx & masky
        # interpolate the with phantom range
        phant_used[mask] = phant_interp((x_new[mask], y_new[mask]))
        # phant_used = phant_interp((x_new, y_new))
        del x_old, y_old, x_new, y_new, phant_interp, maskx, masky, mask
        complex_refractive_index_term = np.tile(np.expand_dims(np.transpose(complex_refractive_index_term),axis=2), (1, nP_x, nP_y))
        phant_used = np.multiply(phant_used, complex_refractive_index_term)
        
        ## fresnel propogate
        if dz == 0:
            dz = np.size(phant.phantom, 0) * phant.dx
        
        if propa_type == 'fresnel':
            if gpu_flag:
                dz = cp.asarray(dz)
                lambda_ = cp.asarray(lambda_)
                fx = cp.asarray(fx)
                fy = cp.asarray(fy)
                phant_used = cp.asarray(phant_used)
                Wi = cp.asarray(Wi)
                gpu_memory('multi_slice Propagation',logger)

                H = cp.exp(cp.multiply(cp.multiply(cp.multiply(-1j*cp.pi, lambda_), dz), fx**2 + fy**2) + 1j*2*cp.pi/lambda_*dz)
                H = cp.fft.fftshift(H, axes=(1,2))
                Wi = cp.multiply(Wi, cp.exp(cp.multiply(-1j*2*cp.pi/lambda_, phant_used)))
                FWi = cp.fft.fftn(Wi, axes=(1,2))
                FWo = cp.multiply(FWi, H)
                Wo = cp.asnumpy(cp.fft.ifftn(FWo, axes=(1,2)))
                del H, FWi, FWo
            else:
                H = np.exp(np.multiply(np.multiply(np.multiply(-1j*np.pi, lambda_), dz), fx**2 + fy**2) + 1j*2*np.pi/lambda_*dz)
                H = np.fft.fftshift(H, axes=(1,2))
                Wi = np.multiply(Wi, np.exp(np.multiply(- 1j*2*np.pi/lambda_, phant_used)))
                Fwi = np.fft.fftn(Wi, axes=(1,2))
                Fwo = np.multiply(Fwi, H)
                Wo = np.fft.ifftn(Fwo, axes=(1,2))

    return Wo
    