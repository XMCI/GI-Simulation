import sys
import numpy as np
from functions.single_slice_propagation_mod3 import single_slice_propagation_mod3
from functions.fresnel_propagation_poly_v2 import fresnel_propagation_poly

def multi_slice_propagation_mod3(Wi, E, FOV, L, D, type_, d_s2, phant, n_slice, propa_type, dim, gpu_flag, logger, configs): 
    '''
        DESCRIPTION:
        MULTI_SLICE_PROPOGATION_MOD2 wave optics simulation, multi-slice approximation but mod 3
        
        CALL:
            - Wi: input wavefront, DIM 2 spatial dimension, DIM 1 energy dimension
            - E: vector containing energy range
            - FOV: target field of view (since the FOV will change in differentslices), unit: m
            - L: distance between G0 and G1, unit: m
            - D: distance between G1 and G2, unit: m
            - type: 'pointsource' or 'planewave'
            - d_s2: the distance between g2 and sample (the side close to g2),unit: m
            - phant: a struct, containing a slice of the phantom and the material of the phantom and the voxel size of the phantom. The phantom matrix is a binary matrix. The phantom should be located between G1 and G2. If no phant exists, use [] as input
            - n_slice: number of slices. The phantom will be evenly sliced.
            - propa_type: propagation type, 'frensel' or 'projection'.
            - gpu_flag: logical variable whether using the GPU or not.
            - dim: '1D' or '2D', the dimension of the wavefront.
            - logger: logger object for logging information.
        OUTPUT:
            - Wo: output wavefront
        
        Updates:
        2024/10/28 (Longchao Men): add the 2D mode
        2023/05/04 (Peiyuan Guo): add gpu method 
    '''

    z_sample = np.size(phant.phantom, 0) * float(phant.dx)
    if not np.all(np.isfinite([d_s2, z_sample, D])) or d_s2 < 0 or z_sample <= 0 or z_sample + d_s2 > D:
        raise ValueError('The sample must fit between G1 and G2')
    nz = phant.phantom.shape[0]
    if not np.isfinite(n_slice) or int(n_slice) != n_slice or not (1 <= n_slice <= nz):
        raise ValueError('n_slice must be an integer in [1, number of voxels]')
    n_slice = int(n_slice)
    phant_sliced = configs.Struct(dx = phant.dx,
                                  material = phant.material)

    if type_ == 'planewave':
        if propa_type != 'fresnel':
            raise NotImplementedError('Only Fresnel propagation is supported here')
        bounds = np.linspace(0, nz, n_slice + 1, dtype=int)
        Wo = fresnel_propagation_poly(
            Wi, E, FOV, L, D - z_sample - d_s2,
            'planewave', dim, gpu_flag, logger
        )
        for a, b in zip(bounds[:-1], bounds[1:]):
            phant_sliced.phantom = phant.phantom[a:b, ...]
            dz_i = (b - a) * float(phant.dx)
            Wo = single_slice_propagation_mod3(
                Wo, E, FOV, L + D - z_sample - d_s2 + a * float(phant.dx),
                dz_i, 'planewave', phant_sliced, propa_type, dim, gpu_flag, logger
            )
        return fresnel_propagation_poly(
            Wo, E, FOV, L + D - d_s2, d_s2,
            'planewave', dim, gpu_flag, logger
        )
    
    if dim == '1D':
        print("multi_slice propagation...")
        ## pointsource
        if type_ == 'pointsource':
            ## propagate before the sample
            M = (L+D)/L
            M_before = (L+D-z_sample-d_s2) / L
            FOV_before = FOV / M # after correction: FOV/M *M_before/M_before
            if propa_type == 'fresnel' or propa_type == 'quasi-projection':
                Vo = fresnel_propagation_poly(Wi, E, FOV_before, L, (D-z_sample-d_s2)/M_before, 'planewave', dim, gpu_flag, logger)
            # Vo stands for the output equivalent plane wave front. The FOV is smaller than that at the detector. The equivalent propagation distance is also smaller.

            ## propagate inside the sample
            slice_index = np.linspace(0, np.size(phant.phantom, 0), n_slice+1, dtype=int, endpoint=True)
            dz_propagate = z_sample / n_slice
            FOVi = FOV_before * M_before
            for ii in np.arange(0, n_slice):
                phant_sliced.phantom = phant.phantom[slice_index[ii]:slice_index[ii+1], :]

                z_propagate = L + D - z_sample - d_s2 + slice_index[ii] * phant.dx
                dz_propagate = (slice_index[ii + 1] - slice_index[ii]) * float(phant.dx)
                Mi = (z_propagate + dz_propagate) / z_propagate
                Vo = single_slice_propagation_mod3(Vo, E, FOVi, z_propagate, dz_propagate/Mi, 'pointsource', phant_sliced, propa_type, dim, gpu_flag, logger)
                FOVi = FOVi * Mi

            ## propagate outside the sample
            if propa_type == 'fresnel' or propa_type == 'quasi-projection':
                z_propagate = L + D - d_s2
                M_after = (L + D) / z_propagate
                Vo = fresnel_propagation_poly(Vo, E, FOVi, z_propagate, d_s2/M_after, 'planewave', dim, gpu_flag, logger)
            ## reshape the plane wave into point source
            Wo = Vo / M

    
    elif dim == '2D':
        ## pointsource
        if type_ == 'pointsource':
            ## propagate before the sample
            M = (L+D)/L
            M_before = (L+D-z_sample-d_s2) / L
            FOV_before = FOV / M # after correction: FOV/M*M_before/M_before
            if propa_type == 'fresnel' or propa_type == 'quasi-projection':
                Vo = fresnel_propagation_poly(Wi, E, FOV_before, L, (D-z_sample-d_s2)/M_before, 'planewave', dim, gpu_flag, logger)

            ## propagate inside the sample
            slice_index = np.linspace(0, np.size(phant.phantom, 0), n_slice+1, dtype=int, endpoint=True)
            dz_propagate = z_sample / n_slice
            FOVi = FOV_before * M_before
            for ii in np.arange(0, n_slice):
                phant_sliced.phantom = phant.phantom[slice_index[ii]:slice_index[ii+1], :, :]

                z_propagate = L + D - z_sample - d_s2 + slice_index[ii] * phant.dx
                dz_propagate = (slice_index[ii + 1] - slice_index[ii]) * float(phant.dx)
                Mi = (z_propagate + dz_propagate) / z_propagate
                Vo = single_slice_propagation_mod3(Vo, E, FOVi, z_propagate, dz_propagate/Mi, 'pointsource', phant_sliced, propa_type, dim, gpu_flag, logger)
                FOVi = FOVi * Mi

                print("\r", end="")
                print("multi_slice_propagation: {}% ".format(round((ii+1)/n_slice*100), 2), end="")
                sys.stdout.flush()
            print()

            ## propagate outside the sample
            if propa_type == 'fresnel' or propa_type == 'quasi-projection':
                z_propagate = L + D - d_s2
                M_after = (L + D) / z_propagate
                Vo = fresnel_propagation_poly(Vo, E, FOVi, z_propagate, d_s2/M_after, 'planewave', dim, gpu_flag,logger)
            ## reshape the plane wave into point source
            Wo = Vo / M
    
    return Wo
