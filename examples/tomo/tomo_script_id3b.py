#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Choose the path to your CHESS repo."""
import sys
#sys.path.append(
#    '/home/rv43/Documents/Programs/repos/CHESSComputing/ChessAnalysisPipeline')
sys.path.append(
    '/home/rv43/Documents/Programs/repos/CHESSComputing/ChessAnalysisPipeline_main')
#sys.path.append(
#    '/nfs/chess/sw/CHESS-software-releases/repos/prod/ChessAnalysisPipeline')

# System modules
import os
from pprint import pprint
from tempfile import NamedTemporaryFile

# Third party modules
import matplotlib.pyplot as plt
from nexusformat.nexus import nxload
from PIL import Image
import yaml

# Local modules
from CHAP.pipeline import PipelineData
from CHAP.common.processor import MapProcessor
from CHAP.common.reader import SpecReader
from CHAP.tomo.processor import (
    TomoCHESSMapConverter,
    TomoReduceProcessor,
    TomoFindCenterProcessor,
    TomoReconstructProcessor,
)
from CHAP.utils.general import quick_imshow

#------------------------------------------------------------------------------#
# Start user input
#------------------------------------------------------------------------------#

"""Select run_type from
'normal':     Full regular reconstruction
              (like running the default hollow_cube in CHAP)
'one_plane':  Reconstruction of two planes only
'two_planes': Reconstruction of one plane only
'roi':        Reconstruction for a subset or rows
              (find center at first and last row)
"""
#run_type = 'normal'
#run_type = 'one_plane'
#run_type = 'two_planes'
run_type = 'roi'

interactive = True
outputdir = 'output'
construct_chess_map = True
reduce_data = True
find_center = True
reconstruct_data = True

map_name = f'{outputdir}/map.nxs'
chess_map_name = f'{outputdir}/chess_map.nxs'
reduced_data_name = f'{outputdir}/reduced_data.nxs'
reduce_config = {
#    'remove_stripe': {'remove_all_stripe': {}},
}
find_center_name = f'{outputdir}/find_center.yaml'
find_center_config = {
    'gaussian_sigma': 0.05,
#    'remove_stripe_sigma': None,
    'ring_width': 1,
}
reconstructed_data_name = f'{outputdir}/reconstructed_data.nxs'
reconstruct_config = {
    'x_bounds': [15, 390],
    'y_bounds': [25, 380],
    'secondary_iters': 10,
#    'gaussian_sigma': 0.75,
#    'remove_stripe_sigma': None,
    'ring_width': 1,
}
tdf_scan_numbers = 1
tbf_scan_numbers = 2
map_config = {
    'title': 'hollow_cube',
    'station': 'id3b',
    'experiment_type': 'TOMO',
    'sample': {'name': 'hollow_cube'},
    'spec_scans': [
        {'spec_file': 'raw/hollow_cube/hollow_cube',
         'scan_numbers': 3}
    ],
    'independent_dimensions': [
        {'label': 'rotation_angles',
         'units': 'degrees',
         'data_type': 'scan_column',
         'name': 'theta'},
        {'label': 'x_translation',
         'units': 'mm',
         'data_type': 'spec_motor',
         'name': 'GI_samx'},
        {'label': 'z_translation',
         'units': 'mm',
         'data_type': 'spec_motor',
         'name': 'GI_samz'}
    ],
}
detector_config = {'detectors': [{'id': 'sim'}],}
detector_setup = {
    'prefix': 'sim',
    'rows': 40,
    'columns': 400,
    'pixel_size': [0.05, 0.005],
    'lens_magnification': 1.0,
}
remove_stripe_corrections = [
#    'remove_all_stripe',
#    'remove_dead_stripe',
#    'remove_large_stripe',
#    'remove_stripe_based_filtering',
#    'remove_stripe_based_fitting',
#    'remove_stripe_based_interpolation',
#    'remove_stripe_based_sorting',
#    'remove_stripe_fw',
#    'remove_stripe_sf',
#    'remove_stripe_ti',
]

#------------------------------------------------------------------------------#
# Optional user input
#------------------------------------------------------------------------------#

if run_type == 'normal':
    img_row_bounds = [3, 35]
    recon_layer_indices = [11, 28]
elif run_type == 'one_plane':
    img_row_bounds = None
    recon_layer_indices = 15
elif run_type == 'two_planes':
    img_row_bounds = None
    recon_layer_indices = [11, 20]
elif run_type == 'roi':
    img_row_bounds = None
    recon_layer_indices = [11, 20]
else:
    print('Pick valid values for img_row_bounds and recon_layer_indices')
    img_row_bounds = None       # list[int, int], optional
    recon_layer_indices = None  # int | list[int, int], optional
                                # ignored when using img_row_bounds

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

if isinstance(recon_layer_indices, int):
    recon_layer_indices = [recon_layer_indices]
if img_row_bounds is None:
    center_rows = None
    if len(recon_layer_indices)== 1:
        detector_config['roi'] = [
            {'start': recon_layer_indices[0],
             'end': recon_layer_indices[0]+1,
             'step': None},
            None]
    elif len(recon_layer_indices) == 2:
        detector_config['roi'] = [
            {'start': recon_layer_indices[0],
             'end': recon_layer_indices[1]},
            None]
        if run_type == 'two_planes':
            detector_config['roi'][0]['step'] = \
                recon_layer_indices[1]-recon_layer_indices[0]-1
        else:
            detector_config['roi'][0]['step'] = None
    else:
        raise RuntimeError('Invalid value for recon_layer_indices ({recon_layer_indices})')
else:
    assert len(img_row_bounds) == 2
    center_rows = recon_layer_indices
    recon_layer_indices = []

reduce_config['img_row_bounds'] = img_row_bounds
find_center_config['center_rows'] = center_rows

#------------------------------------------------------------------------------#

if construct_chess_map:
    print(f'\nmap_config:')
    pprint(map_config)
if reduce_data:
    print(f'\nreduce_config:')
    pprint(reduce_config)
if find_center:
    print(f'\nfind_center_config:')
    pprint(find_center_config)
if reconstruct_data:
    print(f'\nreconstruct_config:')
    pprint(reconstruct_config)
print(f'\ndetector_config:')
pprint(detector_config)

# Construct the CHESS style tomo map
if not construct_chess_map:
    nxroot = nxload(chess_map_name)
else:
    # Create the map for the tomo stack
    tomo_map = MapProcessor(config=map_config, detector_config=detector_config)

    # Read the map for the tomo stack
    tomofields = tomo_map.process(data=None)
    tomofields.save(map_name, mode='w')

    # Read the dark field
    tdf_spec_reader = SpecReader(
        config={
            'station': tomo_map.config.station,
            'experiment_type': tomo_map.config.experiment_type,
            'sample': tomo_map.config.sample,
            'spec_scans': [
                {'spec_file': tomo_map.config.spec_scans[0].spec_file,
                 'scan_numbers': tdf_scan_numbers}],
        },
        detector_config=detector_config)
    darkfield = tdf_spec_reader.read()

    # Read the bright field
    tbf_spec_reader = SpecReader(
        config={
            'station': tomo_map.config.station,
            'experiment_type': tomo_map.config.experiment_type,
            'sample': tomo_map.config.sample,
            'spec_scans': [
                {'spec_file': tomo_map.config.spec_scans[0].spec_file,
                 'scan_numbers': tbf_scan_numbers}],
        },
        detector_config=detector_config)
    brightfield = tbf_spec_reader.read()

    # Convert to CHESS style tomography map
    data = [
        PipelineData(
            name='MapProcessor', data=tomofields, schema='tomofields'),
        PipelineData(name='SpecReader', data=darkfield, schema='darkfield'),
        PipelineData(
            name='SpecReader', data=brightfield, schema='brightfield'),
        PipelineData(
            name='YAMLReader', data=detector_setup,
            schema='tomo.models.Detector')
    ]
    chess_map = TomoCHESSMapConverter()
    _, _, nxroot = chess_map.process(data=data)
    nxroot = nxroot['data']
    nxroot.save(chess_map_name, mode='w')

# Reduce the data
if not reduce_data:
    nxroot = nxload(reduced_data_name)
else:
    nxroot = nxload(chess_map_name)
    # Reduce the data with remove_all_stripe
    data = [
        PipelineData(name='TomoCHESSMapConverter', data=nxroot, schema=None)]
    tomo = TomoReduceProcessor(
        config=reduce_config, interactive=interactive)
    (metadata, provenance, images, reduced_data) = tomo.process(data)
    reduced_data = reduced_data['data']
    reduced_data.save(reduced_data_name, mode='w')

#    for (buf, ext), name in images.get('data', []):
#        buf.seek(0)
#        plt.imshow(Image.open(buf))
#        plt.axis('off')
#        plt.tight_layout()
#        plt.show(block=True)
#        buf.close()
#        plt.close()
    nxentry = reduced_data[reduced_data.default]
    nxdata = nxentry[nxentry.default]
    image_slice = nxdata.nxsignal[0,:,0,:]
    vmin = image_slice.min()
    vmax = image_slice.max()
    if len(recon_layer_indices):
        quick_imshow(
            image_slice,
            title=f'Slice {detector_config["roi"][0]["start"]}',
            cmap='gray',
            colorbar=True,
            vmin=vmin, vmax=vmax,
            save_fig=True,
            block=True)
    if len(recon_layer_indices) == 2:
        quick_imshow(
            image_slice,
            title=f'Slice {detector_config["roi"][0]["end"]}',
            cmap='gray',
            colorbar=True,
            vmin=vmin, vmax=vmax,
            save_fig=True,
            block=True)

    # Reduce the data with various stripe removal types
    for method in remove_stripe_corrections:

        data = [PipelineData(
            name='TomoCHESSMapConverter', data=nxroot, schema=None)]
        tomo = TomoReduceProcessor(
            config=reduce_config.update({'remove_stripe': {method: {}}}),
            interactive=interactive)
        (metadata, provenance, images, reduced_data) = tomo.process(data)
        reduced_data = reduced_data['data']
        reduced_data.save(f'reduced_data_{method}.nxs', mode='w')

        nxentry = reduced_data[reduced_data.default]
        nxdata = nxentry[nxentry.default]
        image_slice = nxdata.nxsignal[0,:,0,:]
        quick_imshow(
            image_slice,
            title=f'Slice {detector_config["roi"][0]["start"]} using {method}',
            cmap='gray',
            #cmap='viridis', interpolation='none',
            colorbar=True,
            vmin=vmin, vmax=vmax,
            save_fig=True,
            block=True)

# Find center
if not find_center:
    pass
else:
    reduced_data = nxload(reduced_data_name)

    data = [PipelineData(
        name='TomoCHESSMapConverter', data=reduced_data, schema=None)]
    tomo = TomoFindCenterProcessor(
        interactive=interactive, config=find_center_config)
    (metadata, provenance, images, center_config) = tomo.process(data)
    center_config = center_config['data']
    with open(find_center_name, 'w') as f:
        yaml.dump(center_config, f, sort_keys=False)

# Reconstruct
if not reconstruct_data:
    #nxroot = nxload(reconstruct_data_name)
    pass
else:
    reduced_data = nxload(reduced_data_name)
    with open(find_center_name) as f:
        center_config = yaml.safe_load(f)

    data = [
        PipelineData(
            name='TomoReduceProcessor', data=reduced_data, schema=None),
#        PipelineData(
#            name='YAMLReader', data=reconstruct_config,
#            schema='tomo.models.TomoReconstructConfig'),
#        PipelineData(
#            name='TomoFindCenterProcessor', data=center_config,
#            schema='tomo.models.TomoFindCenterConfig')
    ]
    from pprint import pprint
    pprint(reconstruct_config)
    tomo = TomoReconstructProcessor(
        config=reconstruct_config,
        center_config=center_config,
        interactive=interactive)
    (metadata, provenance, images, reconstructed_data) = tomo.process(data)
    reconstructed_data = reconstructed_data['data']
    reconstructed_data.save(reconstructed_data_name, mode='w')
