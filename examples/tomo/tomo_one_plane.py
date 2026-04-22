#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import sys
sys.path.append(
    '/home/rv43/Documents/Programs/repos/CHESSComputing/ChessAnalysisPipeline')

# System modules
from pprint import pprint

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

interactive = True
construct_chess_map = True
map_name = 'output/map.nxs'
chess_map_name = 'output/chess_map.nxs'
load_converted_map = True
img_row_bounds = [600, 606] # None, [80, 2080] or [1050, 1251]
recon_layer_index = [600, 1400] # One or two values
                                # ignored when using img_row_bounds
reduce_data = True
reduced_data_name = 'output/reduced_data.nxs'
reduce_config = {
    'remove_stripe': {'remove_all_stripe': {}},
    'img_row_bounds': img_row_bounds,
}
find_center = True
find_center_name = 'output/find_center.yaml'
find_center_config = {
    'gaussian_sigma': 0.75,
#    'remove_stripe_sigma': None,
    'ring_width': 5,
}
reconstruct_data = True
reconstructed_data_name = 'output/reconstructed_data.nxs'
reconstruct_config = {
    'x_bounds': [90, 2470],
    'y_bounds': [600, 2000],
    'secondary_iters:': 25,
    'gaussian_sigma': 0.75,
    'remove_stripe_sigma': None,
    'ring_width': 5,
}
tdf_scan_numbers = 1
tbf_scan_numbers = 2
map_config = {
    'title': 'badran-4510-b',
    'station': 'id3b',
    'experiment_type': 'TOMO',
    'sample': {'name': 'sample_190-1-Edge'},
    'spec_scans': [
        {'spec_file': 'data/sample_190-1-Edge/refine_1',
         'scan_numbers': 3}
    ],
    'independent_dimensions': [
        {'label': 'rotation_angles',
         'units': 'degrees',
         'data_type': 'scan_column',
         'name': 'GI_samphi'},
        {'label': 'x_translation',
         'units': 'mm',
         'data_type': 'spec_motor',
         'name': 'saxx'},
        {'label': 'z_translation',
         'units': 'mm',
         'data_type': 'spec_motor',
         'name': 'saxz'}
    ],
    'presample_intensity': {
        'data_type': 'scan_column',
        'name': 'ic1'
    }
}
detector_config = {'detectors': [{'id': 'andor2'}],}
detector_setup = {
    'prefix': 'andor2',
    'rows': 2160,
    'columns': 2560,
    'pixel_size': [0.0065, 0.0065],
    'lens_magnification': 5.0,
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
# End user input
#------------------------------------------------------------------------------#

if img_row_bounds is None:
    if len(recon_layer_index)== 1:
        detector_config['roi'] = [
            {'start': recon_layer_index,
             'end': recon_layer_index+1,
             'step': None},
            None]
    elif len(recon_layer_index) == 2:
        detector_config['roi'] = [
            {'start': recon_layer_index[0],
             'end': recon_layer_index[1],
             'step': recon_layer_index[1]-recon_layer_index[0]-1},
            None]
    else:
        print('Invalid value for recon_layer_index ({recon_layer_index})')
        exit()
else:
    assert len(img_row_bounds) == 2
    recon_layer_index = []
    detector_config['roi'] = [
        {'start': img_row_bounds[0],
         'end': img_row_bounds[1],
         'step': 1},
        None]

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
    # Load as needed
    if load_converted_map:
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
    if len(recon_layer_index):
        quick_imshow(
            image_slice,
            title=f'Slice {detector_config["roi"][0]["start"]}',
            cmap='gray',
            colorbar=True,
            vmin=vmin, vmax=vmax,
            save_fig=True,
            block=True)
    if len(recon_layer_index) == 2:
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
    tomo = TomoReconstructProcessor(
        config=reconstruct_config,
        center_config=center_config,
        interactive=interactive)
    (metadata, provenance, images, reconstructed_data) = tomo.process(data)
    reconstructed_data = reconstructed_data['data']
    reconstructed_data.save(reconstructed_data_name, mode='w')
