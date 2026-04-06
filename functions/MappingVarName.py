## MappingVarName.m
#
# DESCRIPTION: this function maps the variable name in the configuration
#              file to the variable name in the main function.
#
# CALL: var_out = MappingVarName(var_in)
#   - var_in: the input variable name, string
#   - var_out: the output variable name, string
#
# UPDATES:
#   2021/11/12 (Chengpeng): first version
#
##
    
def MappingVarName(var_in): 
    if var_in == 'FOV':
        var_out = 'FOV'
    elif var_in == 'nP':
        var_out = 'nP'
    elif var_in == 'nSteps':
        var_out = 'nSteps'
    elif var_in == 'nPeriods':
        var_out = 'nPeriods'
    elif var_in == 'totalLength':
        var_out = 'total_length'

    elif var_in == 'L':
        var_out = 'L'
    elif var_in == 'magRatio':
        var_out = 'M'
    elif var_in == 'srcType':
        var_out = 'Source.type'
    elif var_in == 'psfFlag':
        var_out = 'Source.psf_flag'
    elif var_in == 'srcInten':
        var_out = 'Source.intensity'
    elif var_in == 'specRange':
        var_out = 'spec_paras'
    elif var_in == 'g0Period':
        var_out = 'G0.period'
    elif var_in == 'g0DC':
        var_out = 'G0.dc'
    elif var_in == 'g0Material':
        var_out = 'G0.material'
    elif var_in == 'g0Thickness':
        var_out = 'G0.thickness'
    elif var_in == 'g0BaseMaterial':
        var_out = 'G0.base_material'
    elif var_in == 'g0BaseThickness':
        var_out = 'G0.base_thickness'
    elif var_in == 'g1Period':
        var_out = 'G1.period'
    elif var_in == 'g1DC':
        var_out = 'G1.dc'
    elif var_in == 'g1Material':
        var_out = 'G1.material'
    elif var_in == 'g1Thickness':
        var_out = 'G1.thickness'
    elif var_in == 'g1BaseMaterial':
        var_out = 'G1.base_material'
    elif var_in == 'g1BaseThickness':
        var_out = 'G1.base_thickness'
    elif var_in == 'g2Period':
        var_out = 'G2.period'
    elif var_in == 'g2DC':
        var_out = 'G2.dc'
    elif var_in == 'g2Material':
        var_out = 'G2.material'
    elif var_in == 'g2Thickness':
        var_out = 'G2.thickness'
    elif var_in == 'g2BaseMaterial':
        var_out = 'G2.base_material'
    elif var_in == 'g2BaseThickness':
        var_out = 'G2.base_thickness'
    elif var_in == 'detType':
        var_out = 'det_type'
    elif var_in == 'pixelSize':
        var_out = 'pixel_size'
    elif var_in == 'nBits':
        var_out = 'nbits'
    elif var_in == 'chi':
        var_out = 'chi'

    elif var_in == 'Energy':
        var_out = 'E'

    elif var_in == 'phantomFlag':
        var_out = 'phantom_flag'
    elif var_in == 'phantom':
        var_out = 'phant_path'
    elif var_in == 'phantMaterial':
        var_out = 'phant.material'
    elif var_in == 'disSG2':
        var_out = 'd_s2'
    elif var_in == 'nSlice':
        var_out = 'n_slice'

    elif var_in == 'dFG0':
        var_out = 'd_f_g0'
    elif var_in == 'dFG1':
        var_out = 'd_f_g1'
    elif var_in == 'dFG2':
        var_out = 'd_f_g2'
    
    return var_out

def assignment(var_in, var_out):
    return var_out + ' = ' + var_in + '\n'
    