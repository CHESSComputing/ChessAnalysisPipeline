#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A python script version of the CHAP pipeline version of the 
calibration example.

Modify any user choices in the "Start user input" and "Optional user
input" blocks below.

In particular:

* Choose the path to your CHAP repo by setting sys.path.
"""

# System modules
import os
from pprint import pprint
import sys
from tempfile import NamedTemporaryFile

# Third party modules
import yaml

# Local modules
from CHAP.pipeline import PipelineData
from CHAP.common.reader import SpecReader
from CHAP.edd.processor import (
    MCAEnergyCalibrationProcessor,
    MCATthCalibrationProcessor,
)

#------------------------------------------------------------------------------#
# Start user input
#------------------------------------------------------------------------------#

# Choose the path to your CHAP repo
sys.path.append(
    '/home/rv43/Documents/Programs/repos/CHESSComputing/ChessAnalysisPipeline_main')
#sys.path.append(
#    '/nfs/chess/sw/CHESS-software-releases/repos/prod/ChessAnalysisPipeline')

root = None #'examples/edd'
interactive = True
outputdir = 'output'
energy_calib = True
tth_calib = True

if root is None:
    root = os.getcwd()
if outputdir is None:
    outputdir = '.'
if not os.path.isabs(outputdir):
    outputdir = os.path.normpath(
        os.path.realpath(os.path.join(root, outputdir)))

energy_name = f'{outputdir}/energy_calibration_result.yaml'
tth_name = f'{outputdir}/tth_calibration_result.yaml'
spec_file = f'{root}/data/ceo2-5deg-80um-calib/spec.log'
scan_numbers = 1
detector_ids = [0, 21]

#------------------------------------------------------------------------------#
# Optional user input
#------------------------------------------------------------------------------#

spec_config = {
    'station': 'id3a',
    'experiment_type': 'EDD',
    'spec_scans': [
        {'spec_file': spec_file,
         'scan_numbers': scan_numbers}
    ],
}
energy_config = {
    'peak_energies': [34.276, 34.717, 39.255, 40.231],
    'max_peak_index': 1,
    'materials': [
        {'material_name': 'CeO2',
         'lattice_parameters': 5.41153,
         'sgnum': 225}
    ],
}
energy_detector_config = {
    'baseline': True,
    'mask_ranges': [[650, 850]],
}
if detector_ids is not None:
    energy_detector_config['detectors'] = [{'id':id_} for id_ in detector_ids]
tth_config = {'tth_initial_guess': 5.2}
tth_detector_config = {
    'energy_mask_ranges': [[65, 155]],
}

#------------------------------------------------------------------------------#
# End user input
#------------------------------------------------------------------------------#

if not os.path.isdir(outputdir):
    os.makedirs(outputdir)
try:
    NamedTemporaryFile(dir=outputdir)
except Exception as exc:
    raise OSError(
        'Output directory not accessible for writing ({outputdir})') from exc

#------------------------------------------------------------------------------#

print(f'\nspec_config:')
pprint(spec_config)
if energy_calib:
    print(f'\nenergy_config:')
    pprint(energy_config)
    print(f'\nenergy_detector_config:')
    pprint(energy_detector_config)
if tth_calib:
    print(f'\ntth_config:')
    pprint(tth_config)
    print(f'\ntth_detector_config:')
    pprint(tth_detector_config)

# Perform the energy calibration
if energy_calib:
    # Read the calibration data
    nxroot = SpecReader.run(config=spec_config)

    # Perform the energy calibration
    data = [PipelineData(name='SpecReader', data=nxroot)]
    energy_calib_config, images = MCAEnergyCalibrationProcessor.run(
        data=data, config=energy_config, detector_config=energy_detector_config,
        interactive=interactive)

    # Write the energy calibration results
    with open(energy_name, 'w', encoding='utf-8') as f:
        yaml.dump(energy_calib_config, f, sort_keys=False)

# Perform the tth calibration
if tth_calib:
    # Read the energy calibration results
    with open(energy_name, encoding='utf-8') as f:
        energy_calib_config = yaml.safe_load(f)

    # Read the calibration data
    nxroot = SpecReader.run(config=spec_config)

    # Perform the tth calibration
    data = [
        PipelineData(
            data=energy_calib_config,
            schema='edd.models.MCAEnergyCalibrationConfig'),
        PipelineData(name='SpecReader', data=nxroot),
    ]
    tth_calib_config, images = MCATthCalibrationProcessor.run(
        data=data, config=tth_config, detector_config=tth_detector_config,
        interactive=interactive)
    with open(tth_name, 'w', encoding='utf-8') as f:
        yaml.dump(tth_calib_config, f, sort_keys=False)

