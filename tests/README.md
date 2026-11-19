# Physics regression checks

These tests cover the R01–R20 corrections from section 3 of the local physics
review. They use the existing procedural interfaces and leave the broader source
coherence, boundary-condition, material-data, and runtime redesigns out of scope.

Run from the repository root in an environment with the project dependencies
and `pytest` installed:

```powershell
python -m pytest -q tests/test_physics_regressions.py -p no:cacheprovider
python -m pytest -q tests/test_local_data_smoke.py -p no:cacheprovider
```

The first suite creates synthetic material tables, spectra, and phantoms in a
temporary directory, including explicit analytic attenuation and phase values.
It does not require the untracked input data. Production modules still import
CuPy/PyCUDA and some CPU paths initialize CUDA; removing those dependencies was
R24, outside this change. The checks were run with a working CUDA installation.

The second suite skips if the local Au/Si/PMMA tables, 60 kVp spectrum, detector
response, or `Sph_40.0um_0.4_40.npz` are absent. It preserves the phantom's 120 nm
sampling and all 40 slices, taking a central 200×200 crop for a bounded runtime.
This crop checks execution and CPU/GPU consistency, not finite-FOV convergence
or quantitative agreement with dark-field theory. All outputs remain in pytest
temporary directories; existing simulation files are not overwritten.

## Implemented choices

| Review items | Implementation and verification |
|---|---|
| R01, R08 | Square-root spectral amplitude weights, per-energy ER selection, energy-axis validation. Full polychromatic PSC agrees with independently weighted monochromatic runs in 1D/2D and both detector modes. |
| R02 | G2 allocation occurs once, independent of energy block size. Checked on CPU/GPU with 12 channels and blocks 1/5/12. |
| R03, R04 | `fftfreq` grids, `ifftshift` kernels, endpoint-free coordinates, matching FOV. Common vacuum carrier phase omitted: fields are envelopes. Analytic positive/negative modes, power, zero-distance, and inverse checks cover odd/even grids. |
| R05, R06, R07, R14 | Apply G2 in 1D; use centered cell-overlap source kernels with true material throughput, bar-fraction duty cycle, real convolution, and bounded row chunks. Checked against transparent/opaque limits and 1D/2D stepping agreement. |
| R09 | Both generators save projected voxel counts plus dx; scale_factor restricted to 1; unconverged packing rejected. Actual save expressions are exercised on small sphere phantoms and read back to physical thickness/volume. |
| R10 | Reject mismatched projected slice counts and invalid geometry. Automatic regrouping is deliberately not added; a differently sliced file must preserve the same material volume. |
| R11, R12 | Correct array-axis interpolation; implement common plane-wave multislice; use actual integer slice thickness for point-source scaling. Tests cover slabs, asymmetric features, empty point-source samples, and file-to-output execution. |
| R13 | Use current monochromatic energy and sample exit-to-G2 distance. Background and polychromatic scalar correlation length are NaN. |
| R15, R16 | Distinguish plane/point-source grating matching, rebuild sampling coordinates per iteration, reject unsupported dependency scans and simultaneous coupled-period scans, validate integer parameters. |
| R17 | PhotonCounting retains exact Poisson draws with chi=1. As of 2026-11-19, EnergyIntegral uses user-selected additive Gaussian noise, PSC + N(0, chi²), with chi the fixed standard deviation in PSC units. This is a phenomenological model, not compound Poisson statistics. |
| R18, R19 | Reject fractional pixel bins and noninteger roll steps; use dx for x stepping. Preserve the existing per-sample source-intensity convention. Fixed-fluence refinement must scale srcInten by the cell-size ratio (squared in 2D). |
| R20 | Populate PSC/Phi outputs, use circular phase means, represent unsupported sensitivity by NaN, reject empty cropped ROIs, return NaN visibility at zero signal. |

One additional local correction was necessary to pass R11/R12's uniform-slab
check: nominally identical interpolation boundary coordinates differed by about
1e-21 m and dropped the last material column. An eight-machine-epsilon boundary
tolerance with clipped in-range query coordinates fixes this without widening
the physical phantom by a meaningful distance.

The main simulation's existing output interface uses relative directories, as in
the shipped configs. Its eval-based save operation still does not support arbitrary
Windows absolute paths; that independent runtime issue is not changed here.

## Gaussian energy-integrating noise — 2026-11-19

Both shipped configs enable noiseFlag=1 with chi=1. EnergyIntegral adds independent
zero-mean Gaussian noise after energy summation and pixel integration, without
rounding or clipping; chi is an absolute standard deviation, not a percentage.
Set noiseFlag=0 for noiseless simulations. PhotonCounting still requires chi=1.
The analytic and local-data comparison fixtures explicitly disable noise.

Regression checks cover Gaussian mean/variance at zero and positive signals on
CPU/GPU in 1D/2D, zero and invalid standard deviations, and both noise modes in
monochromatic/polychromatic end-to-end runs.
Validation on 2026-11-19: **102 passed**, no warnings, 12.89 s, using the analytic
regression command above. The large local-data smoke suite was not rerun for this
noise-only change.

## Recorded validation — 2026-11-18

Environment: Python 3.10.14, NumPy 1.26.4, SciPy 1.14.1, CuPy/PyCUDA, NVIDIA
GeForce RTX 4070. Interpreter used locally:
`D:/ProgramData/anaconda3/envs/gi-sim/python.exe`.

- Synthetic/analytic regression suite: **84 passed**, no warnings, 12.93 s.
- Local physical-data smoke suite: **2 passed**, no warnings, 166.31 s.
- Python compilation passed; edited-code whitespace checks passed with Git's
  `cr-at-eol` setting, preserving the existing Windows line endings.

The two smoke cases each run CPU/GPU backgrounds and samples, totaling eight
end-to-end simulations. Observed relative L2 differences of the saved PSC:

| Spectrum | Background | 40-slice sample |
|---|---:|---:|
| Monochromatic 30 keV | 5.02e-16 | 1.43e-14 |
| Polychromatic 1–60 keV | 3.22e-16 | 1.57e-14 |

The full 4000×4000×40 default sample run and a fresh large sphere-packing job were
not executed. Passing these tests does not resolve the broader physical-model
limitations discussed outside section 3. Recompute backgrounds after these
corrections; previous G0/spectral normalization results are not interchangeable.
