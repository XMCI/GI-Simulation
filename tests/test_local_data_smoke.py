"""Optional end-to-end checks against local, untracked XOP/spectrum/phantom data.

These tests skip on a fresh clone without the large/private input files.
They preserve dx and all 40 stored slices, using a 200x200 central crop solely
for CPU/GPU consistency; the crop is not a quantitative dark-field benchmark.
"""
import contextlib
import importlib
import io
import logging
import shutil
import sys
import types
import zipfile
from pathlib import Path

import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from GI_SimDemo_phant_mod5 import GI_SimDemo_phant_mod5

REPO = Path(__file__).resolve().parents[1]
PHANTOM = REPO/'phantom'/'Sph_40.0um_0.4_40.npz'
REQUIRED = [REPO/'XOPDATA'/f'{name}Data.mat' for name in ['Au', 'Si', 'PMMA']]
REQUIRED += [REPO/'spectrums'/'60kVp_spec.mat', REPO/'spectrums'/'ER_60.mat', PHANTOM]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in REQUIRED), reason='Local physical input files are not distributed')


@pytest.fixture(scope='module')
def physical_inputs(tmp_path_factory):
    work = tmp_path_factory.mktemp('local_physical_inputs')
    for folder in ['XOPDATA', 'spectrums', 'phantom']:
        (work/folder).mkdir()
    for source in REQUIRED[:-1]:
        shutil.copyfile(source, work/source.relative_to(REPO))
    with zipfile.ZipFile(PHANTOM) as archive, archive.open('Sph.npy') as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        assert not fortran
        crop = np.empty((shape[0], 200, 200), dtype=dtype)
        row, col = shape[1]//2, shape[2]//2
        for index in range(shape[0]):
            plane = np.frombuffer(stream.read(shape[1]*shape[2]*dtype.itemsize), dtype=dtype).reshape(shape[1:])
            crop[index] = plane[row-100:row+100, col-100:col+100]
    with np.load(PHANTOM) as source:
        metadata = {k: source[k] for k in ['dx', 'gap', 'slice_thickness', 'N_slice']}
    np.savez(work/'phantom'/'crop.npz', Sph=crop, **metadata)
    return work, metadata


@pytest.mark.parametrize('polychromatic', [False, True])
def test_actual_data_background_and_sample_cpu_gpu(physical_inputs, monkeypatch, polychromatic):
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('CUDA device unavailable')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA device unavailable')
    work, metadata = physical_inputs
    monkeypatch.chdir(work)
    base = importlib.import_module('configs.config_bg')
    outputs = {}
    for device in ['CPU', 'GPU']:
        for has_sample in [False, True]:
            name = f'_physical_{device}_{has_sample}_{polychromatic}'
            config = types.ModuleType(name)
            for k, v in vars(base).items():
                if not k.startswith('__'):
                    setattr(config, k, v)
            config.nP = 200
            config.FOV = config.nP*float(metadata['dx'])
            config.pixelSize = config.FOV/4
            config.phantomFlag = int(has_sample)
            config.phantom = 'crop'
            config.nSlice = int(metadata['N_slice'])
            config.disSG2 = .3
            config.useDevice = device
            config.noiseFlag = 0  # Compare deterministic CPU/GPU propagation.
            config.outputDir, config.logDir = 'output/', 'logs/'
            config.outputFilename = name
            config.outputParasNames = 'PSC_Vis_Amp_Phi_Sen_AutoCorrLen'
            if polychromatic:
                config.dynamicParasNames = 'srcInten'
                config.dynamicRange1 = [config.srcInten]
            sys.modules[name] = config
            before = list(logging.getLogger().handlers)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    GI_SimDemo_phant_mod5(name)
                with np.load(work/'output'/(name+'.npz'), allow_pickle=True) as saved:
                    outputs[device, has_sample] = {k: saved[k].copy() for k in saved.files}
            finally:
                del sys.modules[name]
                for handler in list(logging.getLogger().handlers):
                    if handler not in before:
                        handler.close()
                        logging.getLogger().removeHandler(handler)
            for key in ['Vis', 'Amp', 'Phi']:
                assert np.all(np.isfinite(outputs[device, has_sample][key]))
            assert np.all(outputs[device, has_sample]['PSC'][0, 0] >= 0)
    for has_sample in [False, True]:
        cpu, gpu = outputs['CPU', has_sample], outputs['GPU', has_sample]
        assert_allclose(cpu['PSC'][0, 0], gpu['PSC'][0, 0], rtol=2e-6, atol=1e-6)
        assert_allclose(cpu['Vis'], gpu['Vis'], rtol=2e-6, atol=1e-8)
        if has_sample and not polychromatic:
            assert_allclose(cpu['AutoCorrLen'], 1.239842e-9/30*.3/2.4e-6)
        else:
            assert np.isnan(cpu['AutoCorrLen']).all()
