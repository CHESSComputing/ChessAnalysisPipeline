"""pytest-style unittests for CHAP.tomo.processor module."""

# System modules
import os
import shutil

# Third party modules
import pytest

# Local modules
from CHAP.pipeline import (
    PipelineData,
    PipelineItem,
)
from CHAP.common.processor import MapProcessor
from CHAP.common.reader import (
    SpecReader,
    YAMLReader,
)
from CHAP.common.writer import FileTreeWriter
from CHAP.tomo.processor import *


class TestTomo:

    def test_id3b(self):
        run_config = {
            'root': 'CHAP/test/tomo',
            'inputdir': 'input',
            'interactive': False,
            'log_level': 'WARNING'}
        map_config = YAMLReader.run(
            filename='map_id3b.yaml', **run_config)
        sim_config = YAMLReader.run(
            filename='tomo_sim_id3b.yaml', **run_config)
        assert map_config['station'] == sim_config['station']
        detector_config = YAMLReader.run(
            filename='detector_cube.yaml', **run_config)

        simfield = TomoSimFieldProcessor.run(
            config=sim_config, detector_config=detector_config, **run_config)
        data = [PipelineData(
            name='TomoSimFieldProcessor',
            data=simfield,
            schema='tomo.models.TomoSimField')]

        darkfield = TomoDarkFieldProcessor.run(data=data, **run_config)
        data.append(PipelineData(
            name='TomoDarkFieldProcessor',
            data=darkfield,
            schema='tomo.models.TomoDarkField'))

        brightfield = TomoBrightFieldProcessor.run(
            data=data, num_image=10, **run_config)
        data.append(PipelineData(
            name='TomoBrightFieldProcessor',
            data=brightfield,
            schema='tomo.models.TomoBrightField'))

        tomospec = TomoSpecProcessor.run(data=data, **run_config)
        FileTreeWriter.run(
            data=[PipelineData(data=tomospec)],
            force_overwrite=True,
            outputdir='raw/hollow_cube',
            **run_config)

        map_hollow_cube = MapProcessor.run(
            config=map_config, detector_config=detector_config, **run_config)
        data = [PipelineData(
            name='MapProcessor', data=map_hollow_cube, schema='tomofields')]

        darkfield = SpecReader.run(
            config={
                'station': map_config['station'],
                'experiment_type': map_config['experiment_type'],
                'sample': map_config['sample'],
                'spec_scans': [
                    {'spec_file': map_config['spec_scans'][0].spec_file,
                     'scan_numbers': 1}],
            },
            detector_config=detector_config,
            **run_config)
        data.append(PipelineData(
            name='SpecReader', data=darkfield, schema='darkfield'))

        brightfield = SpecReader.run(
            config={
                'station': map_config['station'],
                'experiment_type': map_config['experiment_type'],
                'sample': map_config['sample'],
                'spec_scans': [
                    {'spec_file': map_config['spec_scans'][0].spec_file,
                     'scan_numbers': 2}],
            },
            detector_config=detector_config,
            **run_config)
        data.append(PipelineData(
            name='SpecReader', data=brightfield, schema='brightfield'))

        data = list(TomoCHESSMapConverter.run(data=data, **run_config))

        reduce_config = YAMLReader.run(
            filename='reduce_data_id3b.yaml', **run_config)
        data += list(TomoReduceProcessor.run(
            data=data, config=reduce_config, save_figures=False, **run_config))

        center_config = YAMLReader.run(
            filename='find_center_id3b.yaml', **run_config)
        data += TomoFindCenterProcessor.run(
            data=data, config=center_config, save_figures=False, **run_config)

        recon_config = YAMLReader.run(
            filename='reconstruct_data.yaml', **run_config)
        data += TomoReconstructProcessor.run(
            data=data, config=recon_config, save_figures=False, **run_config)

        tomodata = PipelineItem.get_data(data, schema='tomodata')
        nxentry = tomodata[tomodata.default]
        nxdata = nxentry[nxentry.default]
        reconstructed_data = nxdata.nxsignal
        assert reconstructed_data.shape == (
                reduce_config['img_row_bounds'][1] -
                    reduce_config['img_row_bounds'][0],
                recon_config['y_bounds'][1] -
                    recon_config['y_bounds'][0],
                recon_config['x_bounds'][1] -
                    recon_config['x_bounds'][0])
        assert pytest.approx(reconstructed_data.sum()) == 164.28904724121094

        metadata = PipelineItem.get_data(
            data, schema='foxden.reader.FoxdenMetadataReader')
        user_metadata = {
            'findcenter': TomoFindCenterConfig(
                center_offsets=[-0.5, -0.5], center_stack_index=0,
                **center_config).model_dump(),
            'reconstructed_data': TomoReconstructConfig(
                z_bounds=[
                    0,
                    reduce_config['img_row_bounds'][1] - 
                        reduce_config['img_row_bounds'][0]],
                **recon_config).model_dump(),
            'reduced_data': TomoReduceConfig(**reduce_config).model_dump(),
        }
        assert metadata == {
            'btr': 'unknown',
            'did': '/workflow=tomo_reconstruct',
            'parent_did': None,
            'schema': 'user',
            'user_metadata': user_metadata}

        provenance = PipelineItem.get_data(
            data, schema='foxden.reader.FoxdenProvenanceReader')
        assert provenance == {
            'did': '/workflow=tomo_reconstruct',
            'input_files': [{'name': 'todo.fix: pipeline.yaml'}],
            'parent_did': None,
        }
