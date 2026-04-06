import numpy as np

# Talbot-Lau system parameters
E = 30 # the design energy of the grating interferometer [unit: keV]
lambda_ = 1.239842e-9 / E  # wavelength of the design energy [unit: m]
disSG2 = 0.75 # distance between sample (the side close to g2) and g2 [unit: m]
g2Period = 2.4e-6        # pitch of G2 [unit: m]
g1Period = 4.8e-6        # pitch of G1 [unit: m]
xi = lambda_*disSG2/g2Period
print('the autocorrelation length is: ' + f"{xi*10**6:.2f}" + ' um')
type_ = 'pi'  # type of G1 grating, 'Absorption' or 'pi-phase' or 'pi/2-phase'
Talbot_order = 5
if type_ == 'pi' or type_ == 'Absorption':
    eta = 2
elif type_ == 'pi/2':
    eta = 1

D = Talbot_order * 1 / 2 * g1Period ** 2 / eta ** 2 / lambda_ # distance between G1 and G2 [unit: m]
print('the distance between G1 and G2 is: ' + f"{D:.2e}" + ' m')

np_grating = 20  # number of pixels in one grating period
print('the number of point in one grating period is: ' + f"{np_grating}")
num_periods_perPixel = 50 # number of periods per pixel
num_periods_wholeFOV = 200 # number of periods in the whole FOV

dx = g2Period/np_grating              # spatial resolution [unit: m]
print('the spatial resolution is: ' + f"{dx:.2e}" + ' m')

nP = num_periods_wholeFOV*np_grating     # ensure integer number of total grating period
print('the number of point in the whole FOV is: ' + f"{nP}")

FOV = nP*dx           # field of view [unit: m]
print('the detetor FOV in x direction is: ' + f"{FOV:.2e}" + ' m')
pixelSize = num_periods_perPixel*g2Period    # detector pixel size of the detector [unit: m]
print("the pixel size is: " + f"{pixelSize:.2e}" + ' m')

numPixels = FOV/pixelSize
print('the detetor pixel num in x direction is: ' + f"{numPixels:.1f}")

# Phantom parameters
radius = 200e-6 # radius of the sphere [unit: m]
ratio_r_pixel = pixelSize*pixelSize/(np.pi*radius**2) # the number of spheres per pixel
print('the number of spheres per pixel is: ' + f"{ratio_r_pixel:.1f}")

## Estimate volume fraction
num_spheres = 17  # number of spheres per slice
n_slice = 1 # number of slices
slice_thickness = 2 * radius # projection thickness per slice [unit: m]
pixelSize = 6e-4
VF = (num_spheres*4*np.pi*radius**3/3) / ((FOV-2*pixelSize)**2*slice_thickness)
print('the estimate volume fraction is: ' + f"{VF:.3f}")

sample_thickness = slice_thickness*n_slice # 改变小球直径会导致样品厚度改变，最终模拟出的暗场需要厚度矫正。
print('the total sample thickness is: ' + f"{sample_thickness:.2e}" + ' m')


