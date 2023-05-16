#!/usr/bin/env python
"""
File       : common/processor_t.py
Description: Unit tests for common/processor.py code
"""

# system modules
import time
import os
from shutil import rmtree
import unittest

# local modules
from CHAP.common import (AsyncProcessor,
                         IntegrationProcessor,
                         IntegrateMapProcessor,
                         MapProcessor,
                         NexusToNumpyProcessor,
                         NexusToXarrayProcessor,
                         PrintProcessor,
                         StrainAnalysisProcessor,
                         XarrayToNexusProcessor,
                         XarrayToNumpyProcessor)
from CHAP.pipeline import PipelineData


test_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'data')


class AsyncProcessorTest(unittest.TestCase):
    """Unit test for CHAP.common.AsyncProcessor class"""

    def setUp(self):
        self.processor = AsyncProcessor(PrintProcessor())
        self.data = ['doc0', 'doc1', 'doc2']

    def testProcessor(self):
        """Unit test to test processor"""
        data = self.processor.process(self.data)
        self.assertIsNone(data)


class IntegrationProcessorTest(unittest.TestCase):
    """Unit test for CHAP.common.IntegrationProcessor class"""

    def setUp(self):
        from pyFAI.test.utilstest import create_fake_data
        self.processor = IntegrationProcessor()
        detector_data, integrator = create_fake_data()
        integration_method = integrator.integrate1d
        integration_kwargs = {'npt': 50}
        self.data = (detector_data, integration_method, integration_kwargs)

    def testProcessor(self):
        """Unit test to test processor"""
        from pyFAI.containers import IntegrateResult
        data = self.processor.process(self.data)
        self.assertIsInstance(data, IntegrateResult)


class MapProcessorTest(unittest.TestCase):
    """Unit test for CHAP.common.MapProcessor class"""

    def setUp(self):
        """Create a fake spec file and map configuration"""
        self.spec_file = os.path.join(test_data_dir, 'test_scans')
        map_config = {'title': 'test_map',
                      'station': 'id3b',
                      'experiment_type': 'SAXSWAXS',
                      'sample': {'name': 'test_sample'},
                      'spec_scans': [
                          {'spec_file': self.spec_file,
                           'scan_numbers': [1, 2, 3]}
                      ],
                      'independent_dimensions': [
                          {'label': 'dim_0',
                           'data_type': 'spec_motor',
                           'units': 'units_0',
                           'name': 'mtr_0'},
                          {'label': 'dim_1',
                           'data_type': 'spec_motor',
                           'units': 'units_1',
                           'name': 'mtr_1'}],
                      'scalar_data': [
                          {'label': 'counter_0',
                           'data_type': 'scan_column',
                           'units': 'counts',
                           'name': 'counter_0'}]}
        from datetime import datetime
        from random import random
        dims = map_config['independent_dimensions']
        scalars = map_config['scalar_data']
        scan_numbers = map_config['spec_scans'][0]['scan_numbers']
        dim_values = [-1, 0, 1]
        scan_command = ('ascan'
                        + f' {dims[1]["name"]}'
                        + f' {dim_values[0]}'
                        + f' {dim_values[-1]}'
                        + f' {len(dim_values) - 1}'
                        + ' 1')
        column_labels = f'{dims[1]["name"]}  {"  ".join([s["name"] for s in scalars])}'
        spec_lines = [f'#F {map_config["spec_scans"][0]["spec_file"]}']
        spec_lines += [f'#E {int(datetime.now().timestamp())}']
        spec_lines += [f'#O0 {dims[0]["name"]}']
        spec_lines += [f'#o0 {dims[0]["name"]}']
        for scan_no, dim_0_val in zip(scan_numbers, dim_values):
            spec_lines += [f'#S {scan_no} {scan_command}']
            spec_lines += [f'#P0 {dim_0_val}']
            spec_lines += [f'#L  {column_labels}']
            for dim_1_val in dim_values:
                spec_lines += [f'{dim_1_val} {" ".join([str(random()) for s in scalars])}']
        self.spec_file = os.path.join(test_data_dir,
                                      map_config['spec_scans'][0]['spec_file'])
        with open(self.spec_file, 'w') as specf:
            specf.write('\n'.join(spec_lines))

        self.processor = MapProcessor()
        self.data = [PipelineData(schema='MapConfig',
                                  data=map_config)]

    def testProcessor(self):
        """Unit test to test processor"""
        from nexusformat.nexus import NXentry
        data = self.processor.process(self.data)
        self.assertIsInstance(data, NXentry)

    def tearDown(self):
        """Remove the fake spec file created in the setUp method."""
        os.remove(self.spec_file)


class IntegrateMapProcessorTest(MapProcessorTest):
    """Unit tets for CHAP.common.IntegrateMapProcessor class"""

    def setUp(self):
        """Create a fake spec file, diffraction data files, a map
        configuration, and an integration configuration
        """
        super().setUp()

        from pyFAI.test.utilstest import create_fake_data
        from fabio.tifimage import TifImage

        self.detector_data_dirs = []
        detector_prefix = 'det'
        for scan_number in range(1, 4):
            detector_data_dir = os.path.join(
                test_data_dir,
                f'{os.path.basename(self.spec_file)}_{scan_number:03d}')
            self.detector_data_dirs.append(detector_data_dir)
            os.mkdir(detector_data_dir)
            for scan_step in range(3):
                data, ai = create_fake_data()
                detector_data_file = os.path.join(
                    detector_data_dir,
                    f'{os.path.basename(self.spec_file)}_{detector_prefix}_' \
                    + f'{scan_number:03d}_{scan_step:03d}.tiff')
                TifImage(data=data).write(detector_data_file)

        self.poni_file = os.path.join(test_data_dir, 'det.poni')
        ai.save(self.poni_file)
        integration_config = {'title': 'test_integration',
                              'tool_type': 'integration',
                              'integration_type': 'azimuthal',
                              'detectors': [
                                  {'prefix': detector_prefix,
                                   'poni_file': self.poni_file}
                              ],
                              'radial_min': 0.0,
                              'radial_max': 0.6}
        self.processor = IntegrateMapProcessor()
        self.data += [PipelineData(schema='IntegrationConfig',
                                   data=integration_config)]

    def testProcessor(self):
        from nexusformat.nexus import NXprocess
        data = self.processor.process(self.data)
        self.assertIsInstance(data, NXprocess)

    def tearDown(self):
        """Remove all the fake data files created in the setUp
        method
        """
        super().tearDown()
        os.remove(self.poni_file)
        for detector_data_dir in self.detector_data_dirs:
            rmtree(detector_data_dir)

class NexusToNumpyProcessorTest(unittest.TestCase):
    """Unit test for CHAP.common.NexusToNumpyProcessor class"""

    def setUp(self):
        from nexusformat.nexus import NXdata, NXfield
        self.processor = NexusToNumpyProcessor()
        self.data = [PipelineData(data=NXdata(signal=NXfield([0, 0]),
                                              axes=(NXfield([0, 1]),)))]

    def testProcessor(self):
        """Unit test to test processor"""
        import numpy as np
        data = self.processor.process(self.data)
        self.assertIsInstance(data, np.ndarray)


class NexusToXarrayProcessorTest(unittest.TestCase):
    """Unit test for CHAP.common.NexusToXarrayProcessor class"""

    def setUp(self):
        from nexusformat.nexus import NXdata, NXfield
        self.processor = NexusToXarrayProcessor()
        self.data = [PipelineData(data=NXdata(signal=NXfield([0, 0]),
                                              axes=(NXfield([0, 1]),)))]

    def testProcessor(self):
        """Unit test to test processor"""
        import xarray as xr
        data = self.processor.process(self.data)
        self.assertIsInstance(data, xr.DataArray)


class XarrayToNexusProcessorTest(unittest.TestCase):
    """Unit test for CHAP.common.XarrayToNexusProcessor class"""

    def setUp(self):
        import xarray as xr
        self.processor = XarrayToNexusProcessor()
        self.data = [PipelineData(data=xr.DataArray())]

    def testProcessor(self):
        """Unit test to test processor"""
        from nexusformat.nexus import NXdata
        data = self.processor.process(self.data)
        self.assertIsInstance(data, NXdata)


class XarrayToNumpyProcessorTest(unittest.TestCase):
    """Unit test for CHAP.common.XarrayToNumpyProcessor class"""

    def setUp(self):
        import xarray as xr
        self.processor = XarrayToNumpyProcessor()
        self.data = [PipelineData(data=xr.DataArray())]

    def testProcessor(self):
        """Unit test to test processor"""
        import numpy as np
        data = self.processor.process(self.data)
        self.assertIsInstance(data, np.ndarray)


if __name__ == '__main__':
    unittest.main()
