#!/usr/bin/env python
"""
File       : common/processor_t.py
Description: Unit tests for common/processor.py code
"""

# system modules
import time
import os
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
        self.spec_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
            'test_spec_file')
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
        self.spec_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
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
