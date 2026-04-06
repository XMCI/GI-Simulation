import logging
import os
import sys
import time
import scipy.io
import copy
import json

import numpy as np
import cupy as cp
import math

from functions.MappingVarName import *
from functions.configure_log import configure_logging
from functions.configure_output import configure_output
from functions.create_grating_v2 import create_grating
from functions.fresnel_propagation_poly_v2 import fresnel_propagation_poly
from functions.multi_slice_propagation_mod3 import multi_slice_propagation_mod3
from functions.projection_approximation_v2 import projection_approximation
from functions.conv_PSF_v2 import conv_PSF
from functions.phase_stepping import phase_stepping
from functions.calc_FCA import calc_FCA
    
def GI_SimDemo_phant_mod5(config_filename = None): 
    '''
        DESCRIPTION: this function is the example of the main function for GI simulation.

        CALL: 
        - config_filename: the name of configuration file without the file extension '.py', default in the './configs/' subfolder.

        UPDATES:
        2024/10/28 (Longchao Men): add the 2D mode
        2022/09/08 (Peiyuan Guo): add calculate amp and phi for PSC (1D mode)
        2022/07/22 (Peiyuan Guo): set different mode for plane wave and point source (1D mode)
        2022/07/04 (Peiyuan Guo): add phantom (1D mode)
        2021/11/11 (Chengpeng Wu): first version (1D mode)
    '''
    start_time = time.time()
    sys.path.append('configs//')
    configs = __import__(config_filename)
    
    ## Setting output parameters
    para = {} # a dictionary to store all the parameters
    # output parameters
    output_vars, output_num, output_filepath = configure_output(config_filename)
    if output_num == 0:
        raise Exception('Error: the number of output parameters is not equal to 0!')
    # output log file
    logger = configure_logging(config_filename)

    ## Setting constant basic parameters
    dim = configs.propagationDim
    # imaging parameters
    para['FOV'] = configs.FOV
    para['nP'] = configs.nP
    if dim == '1D':
        x = np.linspace(0, para['FOV'], para['nP']).reshape(1,-1)
        dx = x[0,1]-x[0,0]
        coord = configs.Struct(x = x)
    elif dim == '2D':
        x = np.linspace(0, para['FOV'], para['nP']).reshape(1,-1)
        y = np.linspace(0, para['FOV'], para['nP']).reshape(1,-1)
        dx = x[0,1]-x[0,0]
        dy = y[0,1]-y[0,0]
        x, y = np.meshgrid(x, y)
        coord = configs.Struct(x = x, y = y)
    
    para['nSteps'] = configs.nSteps
    para['nPeriods'] = configs.nPeriods
    
    # geometry parameters
    para['total_length'] = configs.totalLength
    para['M'] = configs.magRatio
    para['L'] = para['total_length'] / para['M']
    para['D'] = para['total_length'] - para['L']

    # grating parameters
    para['G0'] = configs.Struct(period = configs.g0Period,
                                dc = configs.g0DC,
                                material = configs.g0Material,
                                thickness = configs.g0Thickness,
                                base_material = configs.g0BaseMaterial,
                                base_thickness = configs.g0BaseThickness)
    
    para['G1'] = configs.Struct(period = configs.g1Period,
                                dc = configs.g1DC,
                                material = configs.g1Material,
                                thickness = configs.g1Thickness,
                                base_material = configs.g1BaseMaterial,
                                base_thickness = configs.g1BaseThickness,
                                type = configs.g1Type)
    
    if para['G1'].type in ['Absorption','pi-phase','pi/2-phase']:
        print('G1 grating is the type of', para['G1'].type)
    else:
        raise Exception('TypeError: specified G1 grating type: (', para['G1'].type, ') is unknown!')
    
    if hasattr(configs, 'g1Structure'):
        para['G1'].structure = configs.g1Structure
        if np.array(['Rectangle','Triangle','Trapezoid']) == para['G1'].structure:
            print('G1 grating is the ', para['G1'].structure, 'structure.')
        else:
            raise Exception('TypeError: specified G1 grating structure: (', para['G1'].structure, ') is unknown!')
    else:
        para['G1'].structure = 'Rectangle'
    if para['G1'].structure == 'Trapezoid':
        para['G1'].top_base = configs.g1TopBase
    
    para['G2'] = configs.Struct(period = configs.g2Period,
                                dc = configs.g2DC,
                                material = configs.g2Material,
                                thickness = configs.g2Thickness,
                                base_material = configs.g2BaseMaterial,
                                base_thickness = configs.g2BaseThickness)

    # source parameters
    para['Source'] = configs.Struct(type = configs.srcType,
                                    psf_flag = configs.psfFlag,
                                    intensity = configs.srcInten)

    # spectrum parameters
    spec_paras = configs.specRange
    para['E'] = np.linspace(spec_paras[0], spec_paras[2], int(math.ceil((spec_paras[2]-spec_paras[0])/spec_paras[1]+1))).reshape(1,-1) # 变成一维数组
    spec_file = configs.specFile
    # load the spectrum file from .mat file and reshape it to 1D array
    Spec_file = scipy.io.loadmat(spec_file)
    Spec = np.reshape(Spec_file['Spec'], (1,-1))

    # detector parameter
    noise_flag = configs.noiseFlag  
    det_type = configs.detType
    para['pixel_size'] = configs.pixelSize
    nbits = configs.nBits
    para['chi'] = configs.chi
    responseFile = configs.responseFile
    # load the detector response file from .mat file and reshape it to 1D array
    ER_file = scipy.io.loadmat(responseFile)
    ER = np.reshape(ER_file['ER'][0, 0:spec_paras[2]], (1,-1))

    if 'EnergyIntegral' == det_type:
        weight = np.multiply(Spec, para['E'])
    else:
        if 'PhotonCounting' == det_type:
            weight = Spec
        else:
            raise Exception('TypeError: the type of detector: (',det_type,') is unknown!')

    weight = weight.T / np.sum(weight.T, axis=0) # normalize the weight
    weight[np.isnan(weight)] = 0 # set the nan value to 0

    E_bar = np.sum(np.multiply(Spec, para['E'])) / np.sum(Spec, axis=1).item()
    lambda_bar = 1.23964181383 / E_bar * 1e-09
    
    # computing device
    useDevice = configs.useDevice
    if 'CPU' == useDevice:
        gpu_flag = False
        print('The simulation is computed by using the CPU')
        logger.info('The simulation is computed by using the CPU')
    else:
        if 'GPU' == useDevice:
            if cp.cuda.Device():
                gpu_flag = True
                device_count = cp.cuda.runtime.getDeviceCount()
                print('The GPU devices num: {}'.format(device_count))
                logger.info('The GPU devices num: {}'.format(device_count))
                device_names = list()
                for i in range(device_count):
                     device_properties = cp.cuda.runtime.getDeviceProperties(i)
                     device_name = device_properties['name']
                     device_names.append(device_name)
                if device_count == 1:
                    print('The simulation is computed by using the GPU: ' + str(device_names[0]))
                    logger.info('The simulation is computed by using the GPU: ' + str(device_names[0]))
                else:
                    print('The simulation is computed by using the GPU: ' + '_'.join(device_name))
                    logger.info('The simulation is computed by using the GPU: ' + '_'.join(device_name))   
            else:
                gpu_flag = False
                print('Warning: this computer has no available GPU device!')
                print('The simulation is computed by using the CPU')
                logger.info('Warning: this computer has no available GPU device!')
                logger.info('The simulation is computed by using the CPU')
        else:
            raise Exception('TypeError: specified device (', useDevice, ') is unknown!')
    
    # propagation type
    propa_type = configs.propagationType
    if propa_type in ['fresnel']:
        print('Simulation the propagation in the type of ' + propa_type)
        logger.info('Simulation the propagation in the type of ' + propa_type)
    else:
        raise Exception('TypeError: specified propagation type (', propa_type, ') is unknown!')
    
    system_type = configs.systemType
    if system_type in ['geometry','Talbot-Lau']:
        print('Simulation the system in the type of', system_type)
        logger.info('Simulation the system in the type of ' + system_type)
    else:
        raise Exception('TypeError: specified system type (', system_type, ') is unknown!')
    
    # phantom parameter
    phantom_flag = configs.phantomFlag
    if phantom_flag:
        para['d_s2'] = configs.disSG2
        para['n_slice'] = configs.nSlice

        print('Loading phantom...')
        phant_name = configs.phantom
        if configs.propaMode == 'mult_slice':
            if os.path.exists('phantom//' + phant_name + '.mat'):
                phant_file = scipy.io.loadmat('phantom//' + phant_name + '.mat')
                phant = configs.Struct(dx = phant_file['p2d'],
                                       phantom = phant_file['phantom'],
                                       material = configs.phantomMaterial)
            elif os.path.exists('phantom//' + phant_name + '.npz'):
                phant_file = np.load('phantom//' + phant_name + '.npz')
                phant = configs.Struct(dx = phant_file['p2d'],
                                       phantom = phant_file['phantom'],
                                       material = configs.phantomMaterial)

        elif configs.propaMode == 'projection_approxi': # now just for sphere phantom
            phant_file = np.load('phantom//' + phant_name + '.npz')
            phant = configs.Struct(phantom = phant_file['Sph'], # (n_slice, nP, nP)
                                   gap = phant_file['gap'],
                                   dx = phant_file['dx'],
                                   slice_thickness = phant_file['slice_thickness'],
                                   material = configs.phantomMaterial)
                              
        else:
            raise Exception('Error: the phantom file: ', phant_name, 'does not exist!')
        

    ## Setting dynamic parameters
    nDynamicParas = configs.nDynamicParas
    dynamicNames = configs.dynamicParasNames
    dynamicNames = str.split(dynamicNames,'_')
    assert(len(dynamicNames) == nDynamicParas), 'Error: the number of dynamic paramter names does not equal to nDynamicParas!'
    dynamic_ranges = np.zeros((nDynamicParas, 3))
    dynamic_vars = np.empty([1, nDynamicParas], dtype=object)
    dynamic_lens = np.zeros((1, nDynamicParas), dtype=int)
    
    for ii in np.arange(0, nDynamicParas):
        field_name = 'dynamicRange' + str(ii+1)
        exec('dynamic_ranges[ii, :] = configs.'+ field_name)

        # the dimension of every dynamic variable is : N x 1
        dynamic_vars[0, ii] = np.transpose((np.linspace(dynamic_ranges[ii,0], dynamic_ranges[ii,2], int(math.ceil((dynamic_ranges[ii,2]-dynamic_ranges[ii,0])/dynamic_ranges[ii,1]+1)))).reshape(1,-1))
        
        # the length of every dynamic variable is N
        dynamic_lens[0, ii] = np.size(dynamic_vars[0, ii], axis=0)
    
    ## Reshaping the dynamic parameters from N dimension to 1 Dimension
    reshape_vars = np.empty([1, nDynamicParas], dtype=object)
    for ii in np.arange(0, nDynamicParas):
        tmp_shape = dynamic_lens.reshape(-1,1)

        # Step 1: remove the ii-th dimension size
        tmp_shape = np.delete(tmp_shape, ii)

        # Step 2: add the first dimension as 1
        tmp_shape = np.insert(tmp_shape, 0, 1).astype(int)

        # Step 3: generate the repeated matrix with the dimension: [N_ii, N_1, N_2, ..., N_(ii-1), N_(ii+1), ..., N_L]
        repeated_matrix = dynamic_vars[0, ii]
        if nDynamicParas > 1: 
            repeated_matrix = np.repeat(repeated_matrix, tmp_shape[1], axis=1)
        if nDynamicParas > 2: 
            for iii in np.arange(2, nDynamicParas):
                repeated_matrix = np.expand_dims(repeated_matrix, axis=iii)
                repeated_matrix = np.repeat(repeated_matrix, tmp_shape[iii], axis=iii)
        reshape_vars[0, ii] = repeated_matrix

        # Step 4: reorder the dimension of the matrix as : [N_1, N_2, ..., N_(ii-1), N_ii, N_(ii+1), ..., N_L]
        tmp_order = np.arange(0, np.size(dynamic_lens))
        tmp_order[np.arange(0, ii)] = np.arange(1, ii+1)
        tmp_order[ii] = 0
        if np.size(tmp_order) != 1:
            reshape_vars[0, ii] = np.transpose(reshape_vars[0, ii], tmp_order)

        # Step 5: reshape the matrix to the dimension as: [prod(N), 1]
        reshape_vars[0, ii] = np.reshape(reshape_vars[0, ii], [int(np.prod(dynamic_lens)), -1], order="F")

    
    ## Simulation
    # the output mean results; shape: (1, prod(dynamic_lens))
    PSC = np.empty((1, np.prod(dynamic_lens).astype(int)), dtype=object)
    Vis = np.zeros((1, np.prod(dynamic_lens).astype(int)))
    VisErr = np.zeros((1, np.prod(dynamic_lens).astype(int)))
    Amp = np.zeros((1, np.prod(dynamic_lens).astype(int)))
    AmpErr = np.zeros((1, np.prod(dynamic_lens).astype(int)))
    Phi = np.zeros((1, np.prod(dynamic_lens).astype(int)))
    Sen = np.zeros((1, np.prod(dynamic_lens).astype(int)))
    AutoCorrLen = np.zeros((1, np.prod(dynamic_lens).astype(int)))

    visTotal = np.empty((1, np.prod(dynamic_lens).astype(int)), dtype=object)
    ampTotal = np.empty((1, np.prod(dynamic_lens).astype(int)), dtype=object)
    phiTotal = np.empty((1, np.prod(dynamic_lens).astype(int)), dtype=object)

    
    for ii in np.arange(0, np.prod(dynamic_lens).astype(int)):
        ## Find the corresponding parameters
        for jj in np.arange(0, nDynamicParas):
            dynamic_varname = MappingVarName(dynamicNames[jj])
            if '.' in dynamic_varname:
                dynamic_varname = dynamic_varname.split('.')
                exec('para[dynamic_varname[0]].' + dynamic_varname[1] + '= reshape_vars[0, jj][ii].item()') 
            else:
                para[dynamic_varname] = reshape_vars[0, jj][ii].item()

        ## some parameters may change with dynamic parameters
        para['L'] = para['total_length'] / para['M']
        para['D'] = para['total_length'] - para['L']
        if 'geometry' == system_type:
            if 'g1Period' in dynamicNames:
                para['G2'].period = (1 + para['D'] / para['L']) * para['G1'].period
                para['G0'].period = para['L'] / para['D'] * para['G2'].period
            elif 'g0Period' in dynamicNames:
                para['G2'].period = para['D'] / para['L'] * para['G0'].period
                para['G1'].period = para['L'] / (para['L'] + para['D']) * para['G2'].period
            elif 'g2Period' in dynamicNames:
                para['G0'].period = para['L'] / para['D'] * para['G2'].period
                para['G1'].period = para['L'] / (para['L'] + para['D']) * para['G2'].period

        elif 'Talbot-Lau' == system_type:
            if para['G1'].type == 'pi-phase':
                eta = 2
            elif para['G1'].type == 'pi/2-phase':
                eta = 1
            if 'g1Period' in dynamicNames:
                para['G2'].period = (para['L'] + d) / (eta * para['L']) * para['G1'].period # d is the Talbot distance at design energy of G1, It needs to be defined in advance and now is not considered.
                para['G0'].period = para['L'] / para['D'] * para['G2'].period
            elif 'g0Period' in dynamicNames:
                para['G2'].period = para['D'] / para['L'] * para['G0'].period
                para['G1'].period = (eta * para['L']) / (para['L'] + para['D']) * para['G2'].period
            elif 'g2Period' in dynamicNames:
                para['G0'].period = para['L'] / para['D'] * para['G2'].period
                para['G1'].period = (eta * para['L']) / (para['L'] + para['D']) * para['G2'].period

        ## Main processes
        # initial wavefront
        if dim == '1D':   
            if not isinstance(para['E'], np.ndarray):
                wf0 = np.sqrt(para['Source'].intensity) * np.ones((np.size(para['E']), para['nP'])) # shape: (1, nP)
            else:
                wf0 = weight * np.sqrt(para['Source'].intensity) * np.ones((np.size(para['E']), para['nP'])) # shape: (nE, nP)
        
        elif dim == '2D':
            if not isinstance(para['E'], np.ndarray):
                wf0 = np.sqrt(para['Source'].intensity) * np.ones((np.size(para['E']), para['nP'], para['nP'])) # shape: (1, nP, nP)
            else:
                if ii == 0:
                    weight = np.expand_dims(weight, 2)
                wf0 = weight * np.sqrt(para['Source'].intensity) * np.ones((np.size(para['E']), para['nP'], para['nP'])) # shape: (nE, nP, nP)
        
        # wf0 = 2**nbits * wf0 # set the initial wavefront intensity to the maximum of detector

        # propagate after G1 and just before G2
        if propa_type == 'fresnel':
            if para['Source'].type == 'pointsource':
                coord_G1 = copy.deepcopy(coord)
                coord_G1.multiply(1/para['M'])
            elif para['Source'].type == 'planewave':
                coord_G1 = copy.deepcopy(coord)
            g1 = create_grating(para['G1'], para['E'], coord_G1, dim, gpu_flag, logger)
            wf1 = np.multiply(wf0, g1)
            
            # the part of simulation with phantom
            if phantom_flag:
                if configs.propaMode == 'mult_slice':
                    wf1 = multi_slice_propagation_mod3(wf1, para['E'], para['FOV'], para['L'], para['D'], para['Source'].type, para['d_s2'], phant, para['n_slice'], 'fresnel', dim, gpu_flag, logger, configs)
                elif configs.propaMode == 'projection_approxi':
                    if para['Source'].type == 'pointsource':
                        raise Exception('Error: the projection approximation mode is not supported for point source!')
                    wf1 = projection_approximation(wf1, para['E'], para['FOV'], para['L'], para['D'], para['Source'].type, para['d_s2'], phant, para['n_slice'], 'fresnel', dim, gpu_flag, logger, configs)
            
            # the part of simulation without phantom
            else:
                if para['Source'].type == 'pointsource':
                    wf1 = fresnel_propagation_poly(wf1, para['E'], para['FOV']/para['M'], para['L'], para['D']/para['M'], 'planewave', dim, gpu_flag, logger)
                    wf1 = wf1 / para['M']
                elif para['Source'].type == 'planewave':
                    wf1 = fresnel_propagation_poly(wf1, para['E'], para['FOV'], para['L'], para['D'], 'planewave', dim, gpu_flag, logger)
        
        ### Notice: don't use this part, the projection propagation. or you improve it.
        else:
            if propa_type == 'projection' or propa_type == 'quasi-projection':
                if phantom_flag:
                    if para['Source'].type == 'pointsource':
                        x = x / para['M']
                    g1 = create_grating(para['G1'], para['E'], x, dim, gpu_flag)
                    wf1 = np.multiply(wf0,g1)
                    wf1 = multi_slice_propagation_mod3(wf1, para['E'], para['FOV'], para['L'], para['D'], para['Source'].type, para['d_s2'], phant, para['n_slice'], propa_type, gpu_flag)
                else:
                    if para['Source'].type == 'pointsource':
                        para['G1'].period = (1 + para['D']/para['L']) * para['G1'].period
                        wf0 = wf0 / para['M']
                    g1 = create_grating(para['G1'], para['E'], x, dim, gpu_flag)
                    wf1 = np.multiply(wf0, g1)
        
        # indensity before G2       
        I0 = np.abs(wf1) ** 2
        # np.save('results//I0.npy', I0)

        # convolute with the source psf or not
        if para['Source'].psf_flag:
            # real psf
            I1 = conv_PSF(I0, para['G0'], para['E'], coord, para['L'], para['D'], dim, gpu_flag, logger)
        else:
            # ideal psf
            I1 = I0
        # np.save('results//I1.npy', I1)
        
        # phase-stepping and detection
        PSC_flat = phase_stepping(I1, para['nSteps'], para['nPeriods'], para['E'], coord, para['G2'], ER, para['pixel_size'], para['chi'], dim, gpu_flag, noise_flag, logger)
        # calculate the visibility, sensitivity and autocorrelation length
        amp_all,vis_all,phi_all = calc_FCA(PSC_flat, para['nPeriods'], dim)
        # np.save('results//amp_all.npy', amp_all)
        # np.save('results//phi_all.npy', phi_all)
        # np.save('results//vis_all.npy', vis_all)
        
        
        print('Dynamic No: '+ str(ii+1)+'/'+ str(np.prod(dynamic_lens)))
        logger.info('Dynamic No: '+ str(ii+1)+'/'+ str(np.prod(dynamic_lens)))

        if dim == '1D':
            # to avoid the edge effect, we discard the first and last two points
            Vis[0,ii] = np.mean(vis_all) 
            Amp[0,ii] = np.mean(amp_all)
            Phi[0,ii] = np.mean(phi_all)
            Sen[0,ii] = np.sqrt(2) / Vis[0,ii] / np.sqrt(np.mean(np.sum(PSC_flat, 1)))
            AutoCorrLen[0,ii] = lambda_bar * para['D'] / para['G2'].period
            PSC[0,ii] = PSC_flat
            
            # print some intermediate results
            # print()

        elif dim == '2D':
            # save whole pixels data
            visTotal[0,ii] = vis_all
            ampTotal[0,ii] = amp_all
            phiTotal[0,ii] = phi_all

            # to avoid the edge effect, we discard the first and last two points
            pad_width = 1  
            new_shape = (vis_all.shape[0] - 2 * pad_width, vis_all.shape[1] - 2 * pad_width)

            vis_all = vis_all[pad_width:new_shape[0] + pad_width, pad_width:new_shape[1] + pad_width]
            amp_all = amp_all[pad_width:new_shape[0] + pad_width, pad_width:new_shape[1] + pad_width]

            # calculate the mean value and standard deviation
            Vis[0,ii] = np.mean(vis_all)
            VisErr[0,ii] = np.std(vis_all)
            Amp[0,ii] = np.mean(amp_all)
            AmpErr[0,ii] = np.std(amp_all)
            AutoCorrLen[0,ii] =(1.239842e-9/para['E']) * para['d_s2'] / para['G2'].period

            # print some intermediate results
            # print()
    
    ### save the results you want
    if np.size(dynamic_lens) != 1:
        # reshape the dynamic variables to the original shape
        PSC = np.reshape(PSC, tuple(dynamic_lens[0]), order="F")
        Vis = np.reshape(Vis, tuple(dynamic_lens[0]), order="F")
        Amp = np.reshape(Amp, tuple(dynamic_lens[0]), order="F")
        Phi = np.reshape(Phi, tuple(dynamic_lens[0]), order="F")
        Sen = np.reshape(Sen, tuple(dynamic_lens[0]), order="F")
        AutoCorrLen = np.reshape(AutoCorrLen, tuple(dynamic_lens[0]), order="F")

    ## Output data and log file
    logger.info('the simulation is finished!')
    print('the simulation is finished!')

    logger.info('-------- %s seconds --------' % (time.time() - start_time))
    print('-------- %s seconds --------' % (time.time() - start_time))
    
    logging.shutdown()
    
    # save as npz file
    save_name_list = []
    for ii in np.arange(0, output_num):
        save_name_list.append(output_vars[ii] + ' = ' + output_vars[ii])
    save_name_list = ','.join(save_name_list) # transform the list to string
    eval('np.savez(\'' + output_filepath + '\',' + save_name_list + ')')

    return

    
if __name__ == '__main__':
    # GI_SimDemo_phant_mod5('config_bg')
    GI_SimDemo_phant_mod5('config_sphere')
    