"""Analytic and end-to-end checks for PHYSICS_REVIEW section 3 (R01-R20).

Run from the repository root with: python -m pytest -q tests
Synthetic material/spectrum/phantom files are confined to pytest's temporary dir.
The production modules retain their existing CuPy/PyCUDA dependencies.
"""
import ast
import contextlib
import importlib
import io
import logging
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace as Struct
from unittest.mock import patch

import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.io import savemat

import GI_SimDemo_phant_mod5 as engine
from functions.add_poission_noise import add_poission_noise
from functions.calc_FCA import calc_FCA
from functions.conv_PSF_v2 import conv_PSF, projected_source_kernel, _real_intensity
from functions.create_grating_v2 import create_grating
from functions.detector import detector
from functions.fresnel_propagation_poly_v2 import fresnel_propagation_poly as propagate
from functions.multi_slice_propagation_mod3 import multi_slice_propagation_mod3 as multislice
from functions.phase_stepping import phase_stepping
from functions.projection_approximation_v2 import (
    projection_approximation, fresnel_propagation_in_sample,
)
from functions.single_slice_propagation_mod3 import single_slice_propagation_mod3 as single_slice
from phantom.SphereFlat import compute_thickness_map, relaxation_pack
from phantom.Sphere_3D_v5 import _rasterize_single_sphere, relax_sphere_centers

LOGGER = logging.getLogger('physics-regressions')
REPO = Path(__file__).resolve().parents[1]
HC = 1.239842e-9

try:
    HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except cp.cuda.runtime.CUDARuntimeError:
    HAS_GPU = False
BACKENDS = [False, pytest.param(True, marks=pytest.mark.skipif(not HAS_GPU, reason='CUDA device unavailable'))]


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'XOPDATA').mkdir()
    (tmp_path / 'phantom').mkdir()
    # Constant delta and linear mu make interpolation-independent analytic references.
    energies = np.array([1000., 20000., 40000., 60000.])
    for name, delta, mu in [('Si', 5.4e-7, 3.), ('Au', 3.5e-6, 500.), ('PMMA', 3e-7, .4)]:
        attenuation = mu * (1.5 - energies / 100000.)
        beta = attenuation * 100 * (HC * 1000 / energies) / (4 * np.pi)
        savemat(tmp_path / 'XOPDATA' / (name + 'Data.mat'),
                {'MatData': np.column_stack([energies, np.full(4, delta), beta, attenuation])})
    before = list(logging.getLogger().handlers)
    yield tmp_path
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if handler not in before:
            handler.close()
            root_logger.removeHandler(handler)


def coords(n=80, dx=1.2e-7, dim='2D'):
    x = np.arange(n) * dx
    if dim == '1D':
        return Struct(x=x[None, :])
    x, y = np.meshgrid(x, x)
    return Struct(x=x, y=y)


def grating(**overrides):
    values = dict(period=2.4e-6, dc=.5, material='Au', thickness=40e-6,
                  base_material='Si', base_thickness=250e-6)
    values.update(overrides)
    return Struct(**values)


def call(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


@pytest.mark.parametrize('gpu', BACKENDS)
@pytest.mark.parametrize('n', [63, 64])
@pytest.mark.parametrize('dim', ['1D', '2D'])
def test_free_space_analytic_modes_energy_and_inverse(gpu, n, dim):
    e, fov, z, m = 30., 48e-6, .35, 5
    x = np.arange(n)
    u = (1 + .3 * np.exp(2j*np.pi*m*x/n) + .2*np.exp(-2j*np.pi*m*x/n))[None, :]
    phase = np.exp(-1j*np.pi*(HC/e)*z*(m/fov)**2)
    expected = 1 + (u - 1) * phase
    if dim == '2D':
        u = np.tile(u[:, None, :], (1, n, 1))
        expected = np.tile(expected[:, None, :], (1, n, 1))
    out = call(propagate, u, e, fov, .35, z, 'planewave', dim, gpu, LOGGER)
    assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
    assert_allclose(np.sum(abs(out)**2), np.sum(abs(u)**2), rtol=1e-13)
    assert_allclose(call(propagate, u, e, fov, .35, 0., 'planewave', dim, gpu, LOGGER), u, atol=1e-12)
    assert_allclose(call(propagate, out, e, fov, .35, -z, 'planewave', dim, gpu, LOGGER), u, atol=1e-12)


@pytest.mark.parametrize('gpu', BACKENDS)
def test_all_propagators_use_same_frequency_grid(gpu):
    n, fov, z, e = 31, 31e-6, .002, np.array([[20., 30., 40.]])
    rng = np.random.default_rng(17)
    u = rng.normal(size=(3, n, n)) + 1j*rng.normal(size=(3, n, n))
    expected = call(propagate, u, e, fov, .35, z, 'planewave', '2D', gpu, LOGGER)
    projected = call(fresnel_propagation_in_sample, u, HC/e, fov, z,
                     np.zeros_like(u), 'fresnel', '2D', gpu, LOGGER)
    phantom = Struct(phantom=np.zeros((1, n, n)), dx=fov/n, material='PMMA')
    sliced = call(single_slice, u, e, fov, .35, z, 'planewave', phantom, 'fresnel', '2D', gpu, LOGGER)
    assert_allclose(projected, expected, atol=1e-12)
    assert_allclose(sliced, expected, atol=1e-12)


@pytest.mark.parametrize('gpu', BACKENDS)
def test_g2_energy_blocks_do_not_drop_channels(gpu):
    e = np.arange(20., 32.).reshape(1, -1)
    g, c = grating(), coords(40)
    reference = call(create_grating, g, e, c, '2D', gpu, LOGGER, abs_flag=1, block_size=12)
    for block_size in [1, 5]:
        actual = call(create_grating, g, e, c, '2D', gpu, LOGGER, abs_flag=1, block_size=block_size)
        assert_allclose(actual, reference, rtol=1e-6)
        assert np.all(actual.mean(axis=(1, 2)) > 0)
    assert np.all((reference >= 0) & (reference <= 1))


@pytest.mark.parametrize('n', [63, 64])
@pytest.mark.parametrize('width', [20., 19.7])
@pytest.mark.parametrize('dc', [0., .25, .5, 1.])
def test_source_kernel_throughput_and_center(n, width, dc):
    kernel = projected_source_kernel(n, 1., width, 1-dc, [1., .8], [0., .1])
    assert_allclose(kernel.sum(axis=1), [(1-dc), (1-dc)*.8+dc*.1], atol=1e-15)
    spectrum = np.fft.fft(np.fft.ifftshift(kernel, axes=1), axis=1)
    assert_allclose(spectrum.imag, 0., atol=1e-15)


@pytest.mark.parametrize('gpu', BACKENDS)
@pytest.mark.parametrize('dim', ['1D', '2D'])
def test_source_convolution_transmission_and_transparent_limit(gpu, dim):
    e = np.array([[20., 30., 40.]])
    c = coords(dim=dim)
    shape = (3, 80) if dim == '1D' else (3, 80, 80)
    u = np.ones(shape)
    for dc in [0., .25, .5, 1.]:
        g = grating(dc=dc)
        out = call(conv_PSF, u, g, e, c, .35, .35, dim, gpu, LOGGER, block_size=2)
        gap = np.exp(-3*(1.5-e.flatten()/100)*100*g.base_thickness)
        bar = gap*np.exp(-500*(1.5-e.flatten()/100)*100*g.thickness)
        expected = (1-dc)*gap + dc*bar
        assert_allclose(out, np.broadcast_to(expected.reshape((-1,)+(1,)*(len(shape)-1)), shape), atol=1e-12)
    out = call(conv_PSF, u, grating(thickness=0., base_thickness=0.), e, c, .35, .35, dim, gpu, LOGGER)
    assert_allclose(out, u, atol=1e-12)


def test_source_intensity_rejects_significant_negative_or_imaginary_values():
    for bad in [np.array([-1.+0j]), np.array([1.+.1j]), np.array([np.nan+0j])]:
        with pytest.raises(ValueError):
            _real_intensity(bad)
    assert_allclose(_real_intensity(np.array([1.+0j, -1e-15+0j])), [1., 0.])


@pytest.mark.parametrize('gpu', BACKENDS)
def test_phase_stepping_1d_matches_2d(gpu):
    n, fov = 80, 80*1.2e-7
    c1, c2 = coords(dim='1D'), coords()
    g = grating()
    intensity = 1 + .8*np.cos(2*np.pi*c1.x/g.period)
    p1 = call(phase_stepping, intensity, 10, 1, 30., c1, g, np.ones((1, 1)),
              fov/4, 1., '1D', gpu, 0, LOGGER)
    p2 = call(phase_stepping, np.tile(intensity[:, None, :], (1, n, 1)),
              10, 1, 30., c2, g, np.ones((1, 1)), fov/4, 1., '2D', gpu, 0, LOGGER)
    assert_allclose(p2[:, 0, :]/(n//4), p1.T, rtol=2e-7, atol=1e-6)
    assert np.all(calc_FCA(p1, 1, '1D')[1] > .1)
    assert_allclose(calc_FCA(p2, 1, '2D')[1][0], calc_FCA(p1, 1, '1D')[1], rtol=2e-7)


@pytest.mark.parametrize('steps,period', [(3, 2.4e-6), (10, 2.41e-6), (2.5, 2.4e-6)])
def test_fractional_phase_steps_rejected(steps, period):
    with pytest.raises(ValueError):
        call(phase_stepping, np.ones((1, 80)), steps, 1, 30., coords(dim='1D'),
             grating(period=period), np.ones((1, 1)), 2.4e-6, 1., '1D', False, 0, LOGGER)


def test_detector_energy_channels_and_integer_pixels():
    u = np.ones((1, 8, 8))
    assert_allclose(detector(u, np.array([[.8]]), 8., 4., None, '2D'), 12.8)
    with pytest.raises(ValueError, match='energy channel'):
        detector(u, np.ones((1, 60)), 8., 4., None, '2D')
    for pixel in [0., -1., 3., 8./3]:
        with pytest.raises(ValueError):
            detector(u, np.ones((1, 1)), 8., pixel, None, '2D')
    # At fixed physical fluence, halve the sample spacing and quarter per-point counts.
    coarse = detector(np.ones((1, 8, 8)), np.ones((1, 1)), 1., .5, None, '2D')
    fine = detector(np.full((1, 16, 16), .25), np.ones((1, 1)), 1., .5, None, '2D')
    assert_allclose(coarse, fine)


@pytest.mark.parametrize('gpu', BACKENDS)
def test_asymmetric_single_slice_keeps_axis_order(gpu):
    n, dx, e = 16, 1e-6, 30.
    ph = np.zeros((1, n, n))
    ph[0, 4, 10] = 1.
    out = call(single_slice, np.ones((1, n, n)), e, n*dx, .35, 1e-30,
               'planewave', Struct(phantom=ph, dx=dx, material='PMMA'),
               'fresnel', '2D', gpu, LOGGER)
    assert np.unravel_index(np.argmin(abs(out[0])**2), (n, n)) == (4, 10)
    assert_allclose(abs(out[0, 4, 10])**2, np.exp(-.4*1.2*100*dx), rtol=1e-12)


@pytest.mark.parametrize('gpu', BACKENDS)
@pytest.mark.parametrize('dim', ['1D', '2D'])
def test_multislice_uniform_slab_conserves_material_when_regrouped(gpu, dim):
    n, dx, nz = 16, 1e-6, 5
    shape = (n,) if dim == '1D' else (n, n)
    phantom = Struct(phantom=np.ones((nz,)+shape), dx=dx, material='PMMA')
    u = np.ones((1,)+shape)
    e = 30.
    transmission = np.exp(-1j*2*np.pi/(HC/e)*3e-7*nz*dx - .4*1.2*100*nz*dx/2)
    for count in [1, 2, 5]:
        out = call(multislice, u, e, n*dx, .35, .35, 'planewave', .1,
                   phantom, count, 'fresnel', dim, gpu, LOGGER, Struct(Struct=Struct))
        assert_allclose(out, transmission*u, atol=1e-12)


@pytest.mark.parametrize('dim', ['1D', '2D'])
def test_point_source_empty_sample_matches_direct_propagation(dim):
    n, dx, nz = 32, 1e-6, 5
    fov = n*dx
    u = (1+.2*np.exp(2j*np.pi*3*np.arange(n)/n))[None, :]
    shape = (n,) if dim == '1D' else (n, n)
    if dim == '2D':
        u = np.tile(u[:, None, :], (1, n, 1))
    phantom = Struct(phantom=np.zeros((nz,)+shape), dx=dx, material='PMMA')
    expected = call(propagate, u, 30., fov/2, .35, .35/2, 'planewave', dim, False, LOGGER)/2
    out = call(multislice, u, 30., fov, .35, .35, 'pointsource', .1,
               phantom, 2, 'fresnel', dim, False, LOGGER, Struct(Struct=Struct))
    assert_allclose(out, expected, atol=1e-12)


def projected_phantom(n=16):
    return Struct(phantom=np.arange(1., 5.)[:, None, None]*np.ones((4, n, n)),
                  dx=1e-6, gap=0., slice_thickness=4e-6, material='PMMA')


@pytest.mark.parametrize('gpu', BACKENDS)
def test_projected_slices_preserve_total_thickness_and_validate_count(gpu):
    ph = projected_phantom()
    u = np.ones((1, 16, 16))
    args = (u, 30., 16e-6, .35, .35, 'planewave', .1)
    for count in [1, 2, 5, 2.5]:
        with pytest.raises(ValueError):
            call(projection_approximation, *args, ph, count, 'fresnel', '2D', gpu, LOGGER, None)
    out = call(projection_approximation, *args, ph, 4, 'fresnel', '2D', gpu, LOGGER, None)
    assert_allclose(abs(out)**2, np.exp(-.4*1.2*100*10e-6), atol=1e-12)
    for d_s2 in [-.1, .35]:
        with pytest.raises(ValueError, match='fit between'):
            call(projection_approximation, u, 30., 16e-6, .35, .35, 'planewave', d_s2,
                 ph, 4, 'fresnel', '2D', gpu, LOGGER, None)
    with pytest.raises(ValueError, match='physical FOV'):
        call(projection_approximation, u, 30., 32e-6, .35, .35, 'planewave', .1,
             ph, 4, 'fresnel', '2D', gpu, LOGGER, None)


def test_poisson_low_counts_nonnegative_mean_variance():
    state = np.random.get_state()
    try:
        np.random.seed(123)
        samples = add_poission_noise(np.full(500000, .1), 1.)
    finally:
        np.random.set_state(state)
    assert samples.min() >= 0
    assert np.all(samples == np.floor(samples))
    assert abs(samples.mean()-.1) < .002
    assert abs(samples.var()-.1) < .002
    for invalid in [np.array([-1.]), np.array([np.nan]), np.array([np.inf])]:
        with pytest.raises(ValueError):
            add_poission_noise(invalid, 1.)
    with pytest.raises(ValueError):
        add_poission_noise(np.ones(4), 2.)


@pytest.mark.parametrize('dim', ['1D', '2D'])
def test_fca_cosine_and_zero_signal(dim):
    psc = 10*(1+.3*np.cos(2*np.pi*np.arange(30)/10+.7))
    psc = psc[None, :] if dim == '1D' else psc[:, None, None]
    amp, vis, phase = calc_FCA(psc, 3, dim)
    assert_allclose(amp, 10., atol=1e-12)
    assert_allclose(vis, .3, atol=1e-12)
    assert_allclose(phase, .7, atol=1e-12)
    assert np.all(np.isnan(calc_FCA(np.zeros_like(psc), 3, dim)[1]))


def test_sphere_projected_chord_and_volume():
    radius, side, n = 20e-6, 80e-6, 400
    # The generator's endpoint=False convention puts the sphere center on a sample.
    with np.errstate(invalid='ignore'):
        thickness = compute_thickness_map([[side/2, side/2]], radius, side, (n, n))
    assert_allclose(thickness[n//2, n//2], 2*radius)
    assert_allclose(thickness.sum()*(side/n)**2, 4*np.pi*radius**3/3, rtol=5e-4)
    with pytest.raises(RuntimeError, match='did not converge'):
        relaxation_pack(2, radius, side, max_iter=0)


@pytest.mark.parametrize('generator', ['Sphere_3D_v5.py', 'SphereFlat.py'])
def test_generator_serialization_round_trip_preserves_path_length(generator, data_dir):
    # Execute the actual save expression on a small phantom: the scripts' large
    # hardcoded examples need not run to verify the loader's unit contract.
    side, n, radius = 80e-6, 80, 20e-6
    dx = side/n
    center = np.full(3, side/2)
    if generator == 'Sphere_3D_v5.py':
        x = (np.arange(n)+.5)*dx
        a, b, c, d, e, f, mask = _rasterize_single_sphere(
            (center, radius, np.full(3, side), x, x, x))
        voxels = np.zeros((n, n, n), dtype=np.uint8)
        voxels[a:b, c:d, e:f] = mask
        sph = voxels.sum(axis=0, dtype=np.float64)[None, :, :]
        physical_path = sph*dx
    else:
        with np.errstate(invalid='ignore'):
            physical_path = compute_thickness_map([center[:2]], radius, side, (n, n))[None, :, :]
        sph = physical_path
    tree = ast.parse((REPO/'phantom'/generator).read_text(encoding='utf-8'))
    save_call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == 'savez')
    namespace = dict(np=np, os=os, Sph=sph, scale_factor=1, dx=dx, gap=0.,
                     slice_thickness=side, n_slice=1, group=1, VF=.4,
                     save_dir=str(data_dir/'phantom')+os.sep, save_name='roundtrip.npz')
    eval(compile(ast.Expression(save_call), generator, 'eval'), namespace)
    generated = next((data_dir/'phantom').glob('*.npz'))
    with np.load(generated) as saved:
        assert 'dx' in saved
        assert_allclose(saved['Sph']*saved['dx'], physical_path, rtol=1e-14, atol=1e-20)
        assert_allclose(saved['Sph'].sum()*float(saved['dx'])**3,
                        4*np.pi*radius**3/3, rtol=.005)


def test_3d_nonconverged_packing_is_rejected():
    with pytest.raises(RuntimeError, match='did not converge'):
        call(relax_sphere_centers, 2, 1e-6, [4e-6]*3, max_iter=0,
             verbose=False, num_cores=1)


@pytest.fixture
def run_simulation(data_dir):
    base = importlib.import_module('configs.config_bg')
    serial = 0

    def run(spec=None, response=None, capture=False, **overrides):
        nonlocal serial
        serial += 1
        name = f'_physics_config_{serial}'
        config = types.ModuleType(name)
        for key, value in vars(base).items():
            if not key.startswith('__'):
                setattr(config, key, value)
        config.nP, config.FOV, config.pixelSize = 80, 9.6e-6, 2.4e-6
        config.useDevice = 'CPU'
        # Match the existing relative-path output interface used by shipped configs.
        config.outputDir = 'outputs/'
        config.logDir = 'logs/'
        config.outputFilename = name
        config.outputParasNames = 'PSC_Vis_Amp_Phi_Sen_AutoCorrLen_visTotal_ampTotal_phiTotal'
        config.specRange = [20, 5, 45]
        config.specFile = str(data_dir / f'{name}_spec.mat')
        config.responseFile = str(data_dir / f'{name}_er.mat')
        spec = np.asarray([1, 4, 3, 2, 1, .5] if spec is None else spec)
        response = np.asarray([.9, .8, .7, .6, .5, .4] if response is None else response)
        savemat(config.specFile, {'Spec': spec.reshape(1, -1)})
        savemat(config.responseFile, {'ER': response.reshape(1, -1)})
        for key, value in overrides.items():
            setattr(config, key, value)
        sys.modules[name] = config
        recordings = []
        original = engine.create_grating

        def record_grating(g, energy, coord, dim, gpu, logger, **kwargs):
            recordings.append((g.period, np.shape(coord.x), coord.x.copy()))
            return original(g, energy, coord, dim, gpu, logger, **kwargs)

        try:
            with patch.object(engine, 'create_grating', side_effect=record_grating):
                call(engine.GI_SimDemo_phant_mod5, name)
            with np.load(data_dir / 'outputs' / (name + '.npz'), allow_pickle=True) as saved:
                result = {key: saved[key].copy() for key in saved.files}
            return (result, recordings) if capture else result
        finally:
            del sys.modules[name]
    return run


@pytest.mark.parametrize('dim', ['1D', '2D'])
@pytest.mark.parametrize('det_type', ['PhotonCounting', 'EnergyIntegral'])
def test_polychromatic_pipeline_equals_weighted_monochromatic_psc(run_simulation, dim, det_type):
    spec = np.array([1., 4., 3., 2., 1., .5])
    energies = np.arange(20., 46., 5.)
    poly = run_simulation(spec=spec, propagationDim=dim, detType=det_type,
                          dynamicParasNames='srcInten', dynamicRange1=[100])
    monochromatic = [run_simulation(spec=spec, propagationDim=dim, detType=det_type,
                                    dynamicRange1=[float(e)])['PSC'][0, 0] for e in energies]
    weights = spec*energies if det_type == 'EnergyIntegral' else spec
    weights /= weights.sum()
    expected = sum(w*p for w, p in zip(weights, monochromatic))
    assert_allclose(poly['PSC'][0, 0], expected, rtol=3e-7, atol=1e-6)
    assert np.isnan(poly['AutoCorrLen'][0, 0])
    assert np.all(np.isfinite(poly['Phi']))
    if dim == '2D' or det_type == 'EnergyIntegral':
        assert np.isnan(poly['Sen'][0, 0])


def test_zero_weight_energy_does_not_affect_psc(run_simulation):
    poly = run_simulation(spec=[0, 0, 1, 0, 0, 0], dynamicParasNames='srcInten', dynamicRange1=[100])
    mono = run_simulation(dynamicRange1=[30])
    assert_allclose(poly['PSC'][0, 0], mono['PSC'][0, 0], rtol=1e-12)


def test_mono_response_is_selected_not_summed(run_simulation):
    a = run_simulation(response=[.9, .8, .7, .6, .5, .4])
    b = run_simulation(response=[.1, .2, .7, .3, .2, .1])
    assert_allclose(a['PSC'][0, 0], b['PSC'][0, 0], rtol=1e-12)
    assert np.isnan(a['AutoCorrLen'][0, 0])


@pytest.mark.parametrize('source', ['planewave', 'pointsource'])
@pytest.mark.parametrize('period_name', ['g0Period', 'g1Period', 'g2Period'])
def test_period_scans_follow_geometry(run_simulation, source, period_name):
    value = 4.8e-6 if period_name == 'g1Period' else 2.4e-6
    result, calls = run_simulation(srcType=source, dynamicParasNames=period_name,
                                   dynamicRange1=[value], capture=True)
    mag = 2 if source == 'pointsource' else 1
    expected_g1 = value if period_name == 'g1Period' else 2*value/mag
    assert_allclose(calls[0][0], expected_g1)
    assert np.all(np.isfinite(result['Vis']))


@pytest.mark.parametrize('name,values', [('nP', [80, 80, 160]), ('FOV', [9.6e-6, 9.6e-6, 19.2e-6])])
def test_sampling_scans_rebuild_coordinates(run_simulation, name, values):
    _, calls = run_simulation(dynamicParasNames=name, dynamicRange1=values, capture=True)
    assert len(calls) == 2
    assert calls[0][1] == (80, 80)
    expected_n = 160 if name == 'nP' else 80
    expected_fov = 19.2e-6 if name == 'FOV' else 9.6e-6
    assert calls[1][1] == (expected_n, expected_n)
    assert_allclose(calls[1][2][0, -1], expected_fov*(expected_n-1)/expected_n)


@pytest.mark.parametrize('name', ['L', 'specRange', 'detType', 'phantomFlag', 'phantom', 'phantMaterial'])
def test_unsupported_dynamic_parameters_rejected(run_simulation, name):
    with pytest.raises(NotImplementedError, match='dependency updates'):
        run_simulation(dynamicParasNames=name, dynamicRange1=[1])


def test_invalid_scans_and_noise_models_rejected(run_simulation):
    with pytest.raises(ValueError, match='positive integer'):
        run_simulation(dynamicParasNames='nSteps', dynamicRange1=[2.5])
    with pytest.raises(ValueError, match='one coupled'):
        run_simulation(nDynamicParas=2, dynamicParasNames='g1Period_g2Period',
                       dynamicRange1=[4.8e-6], dynamicRange2=[2.4e-6])
    with pytest.raises(NotImplementedError, match='compound Poisson'):
        run_simulation(noiseFlag=1)
    with pytest.raises(ValueError, match='match the declared energy grid'):
        run_simulation(response=[1., 1.])
    with pytest.raises(ValueError, match='outside the detector'):
        run_simulation(dynamicRange1=[50])
    with pytest.raises(ValueError, match='nonnegative and nonzero'):
        run_simulation(spec=[0, 0, 0, 0, 0, 0])


def test_phantom_pipeline_autocorrelation_and_uniform_transmission(run_simulation, data_dir):
    n, dx, ns = 80, 1.2e-7, 2
    np.savez(data_dir/'phantom'/'slab.npz', Sph=np.ones((ns, n, n)),
             dx=dx, gap=0., slice_thickness=dx, N_slice=ns)
    bg = run_simulation()
    sample = run_simulation(phantomFlag=1, phantom='slab', phantomMaterial='PMMA',
                            nSlice=ns, disSG2=.1)
    assert_allclose(sample['AutoCorrLen'][0, 0], HC/30*.1/2.4e-6)
    assert_allclose(sample['Amp']/bg['Amp'], np.exp(-.4*1.2*100*ns*dx), rtol=1e-11)
    assert_allclose(sample['Vis']/bg['Vis'], 1., atol=1e-11)
    with pytest.raises(ValueError, match='stored slice count'):
        run_simulation(phantomFlag=1, phantom='slab', nSlice=1, disSG2=.1)
    with pytest.raises(ValueError, match='physical FOV'):
        run_simulation(phantomFlag=1, phantom='slab', nSlice=2, FOV=19.2e-6)


@pytest.mark.parametrize('dim', ['1D', '2D'])
def test_pipeline_phase_uses_circular_mean(run_simulation, dim):
    phase = np.tile([np.pi-.1, -np.pi+.1], 2)
    if dim == '2D':
        phase = np.tile(phase, (4, 1))
    with patch.object(engine, 'calc_FCA', return_value=(
            np.ones_like(phase), np.full_like(phase, .3), phase)):
        result = run_simulation(propagationDim=dim)
    assert_allclose(abs(result['Phi'][0, 0]), np.pi)


def test_crop_requires_nonempty_roi(run_simulation):
    with pytest.raises(ValueError, match='remain after edge cropping'):
        run_simulation(pixelSize=4.8e-6)


@pytest.mark.skipif(not HAS_GPU, reason='CUDA device unavailable')
def test_polychromatic_full_pipeline_cpu_gpu(run_simulation):
    cpu = run_simulation(dynamicParasNames='srcInten', dynamicRange1=[100], useDevice='CPU')
    gpu = run_simulation(dynamicParasNames='srcInten', dynamicRange1=[100], useDevice='GPU')
    assert_allclose(cpu['PSC'][0, 0], gpu['PSC'][0, 0], rtol=3e-7, atol=1e-6)


@pytest.mark.parametrize('dim', ['1D', '2D'])
def test_multislice_full_pipeline_with_voxel_file(run_simulation, data_dir, dim):
    n, dx = 80, 1.2e-7
    shape = (5, n) if dim == '1D' else (5, n, n)
    np.savez(data_dir/'phantom'/'voxel.npz', phantom=np.ones(shape), p2d=dx)
    bg = run_simulation(propagationDim=dim)
    slab = run_simulation(propagationDim=dim, phantomFlag=1, propaMode='mult_slice',
                          phantom='voxel', phantomMaterial='PMMA', nSlice=2, disSG2=.1)
    assert_allclose(slab['Amp']/bg['Amp'], np.exp(-.4*1.2*100*5*dx), rtol=1e-10)
    assert_allclose(slab['Vis']/bg['Vis'], 1., atol=1e-10)
