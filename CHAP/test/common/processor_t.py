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
