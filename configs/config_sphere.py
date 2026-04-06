'''
    This file is used to set the configuration parameters for the simulation.
'''

# to realize struct type like in matlab
class Struct:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    def multiply(self, number):
        for key, value in self.__dict__.items():
            setattr(self, key, value*number)

#################### OUTPUT PARAMETERS ############################
outputParasNames = 'visTotal_ampTotal'         # splicing of all ouput parameters separated by '_'
outputDir = 'results//'                   # directory for storing the output file
logDir = 'logs//'                         # directory for storing the logging file, and filename is same as output data filename with the extension as '.txt'
outputFilename = 'sphere'   # output file name, with the extension '.npy' or 'npz'

#################### CONSTANT BASIC PARAMETERS ####################
# Imaging parameters
FOV = 4.8e-4             # field of view [unit: m]
nP = 4000                # number of sampling points
nSteps = 10              # number of phase-stepping steps in one period
nPeriods = 1             # number of periods in phase-stepping

# Geometry parameters
totalLength = 0.7        # total length of the system [unit: m]
magRatio = 2             # magnification ratio (L+D)/L. if srcType = 'planewave', the magRatio = 2 just represents the system is a symmetric system

# Source parameters
srcType = 'planewave'    # type of source, supporting 'pointsource' or 'planewave'
psfFlag = 1              # flag for considering the source psf or not, 1 or 0
srcInten = 100           # intensity of source per sampling point, which effects the noise level of the final PSC

# Spectrum parameters 
specFile = 'spectrums/60kVp_spec.mat'  # file path of the spectrum, variable name as 'Spec'; one dimension array
specRange = [1, 1, 60]      # energy range (start:interval:stop) [unit: kev]

# Grating parameters
# G0
g0Period = 2.4e-6        # pitch of G0 [unit: m]
g0DC = 0.5               # duty cycle of G0
g0Material = 'Au'           # element symbol of etching material
g0Thickness = 47e-6     # thickness of etching material [unit: m]
g0BaseMaterial = 'Si'     # element symbol of base material
g0BaseThickness = 250e-6      # thickness of base material [unit: m]
# G1
g1Period = 4.8e-6         # pitch of G1 [unit: m]
g1DC = 0.5               # duty cycle of G1
g1Material = 'Si'           # symbol of etching material
g1Thickness = 38.4e-6     # thickness of etching material [unit: m]
g1BaseMaterial = 'Si'     # symbol of base material
g1BaseThickness = 250e-6      # thickness of base material [unit: m]
g1Type = 'pi-phase'      # type of G1 grating, 'Absorption' or 'pi-phase' or 'pi/2-phase'
# G2
g2Period = 2.4e-6        # pitch of G2 [unit: m]
g2DC = 0.5               # duty cycle of G2
g2Material = 'Au'           # symbol of etching material
g2Thickness = 40e-6     # thickness of etching material [unit: m]
g2BaseMaterial = 'Si'     # symbol of base material
g2BaseThickness = 250e-6      # thickness of base material [unit: m]

# Detector parameters
detType = 'EnergyIntegral' # type of detector, 'EnergyIntegral' or 'PhotonCounting'
pixelSize = 1.2e-4      # pixel size of the detector [unit: m]
nBits = 16               # number of bits in the detector data (not used)
chi = 1                  # proportionality factor depending on detector properties
responseFile = 'spectrums/ER_60.mat'  # file path of the detector response, variable name as 'ER'; 要与能谱对应！！
noiseFlag = 0           # flag for adding poisson noise, 1 or 0

# Phantom parameters
phantomFlag = 1          # flag for if a phantom is used, 1 or 0
propaMode = 'projection_approxi' # mode of propagation, 'projection_approxi' or 'mult_slice'
phantom = 'Sph_40.0um_0.4_40'           # mat file saved in ./phantom
phantomMaterial = 'PMMA'   # symbol of phantom material
disSG2 = 0.3             # distance between sample (the side close to g2) and g2, unit: m
nSlice = 40            # number of slices, if no slice then 1


#################### DYNAMIC PARAMETERS ####################
nDynamicParas = 1        # number of dynamic parameters
dynamicParasNames = 'Energy'   # splicing of all dynamic numerical parameters separated by '_'
# the value of every dynamic paramter is formatted as 'start:interval:end'.
dynamicRange1 = [30] 
# dynamicRange1 = [1, 1, 60]
# dynamicRange2 = [0.01, 0.02, 0.3]
# dynamicRange1 = [2.4e-5, 2.4e-5, 1.2e-4]
# dynamicRange1 = [10, 10, 40]


useDevice = 'GPU'                        # specified computing device, 'CPU' or 'GPU'
propagationType = 'fresnel'              # specified propagation type, 'fresnel' or 'projection'
systemType = 'Talbot-Lau'                  # specified system type 'geometry' or 'Talbot-Lau'
# Notice: '2D' mod is symmetrical in the x-y direction
propagationDim = '2D'                    # specified propagation dimension, '1D' or '2D'