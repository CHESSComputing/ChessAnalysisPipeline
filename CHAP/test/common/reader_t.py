#!/usr/bin/env python
"""
File       : common/reader_t.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Unit tests for common/reader.py code
"""

# System modules
import os
import unittest

# Local modules
from CHAP.common.reader import (
    BinaryFileReader,
    NexusReader,
    URLReader,
    YAMLReader,
)

test_data_dir = os.path.join(
    os.path.dirname(
        os.path.dirname(__file__)
    ),
    'data'
)

#FIX
# pylint: disable=too-many-function-args
# pylint: disable=unexpected-keyword-arg
class BinaryFileReaderTest(unittest.TestCase):
    """Unit test for CHAP.common.BinaryFileReader class"""

    def setUp(self):
        self.reader = BinaryFileReader(
            filename=os.path.join(test_data_dir, 'img.png'),
        )

    def testReader(self):
        """Unit test to test reader"""
        data = self.reader.read()
        self.assertIsInstance(data, bytes)


class NexusReaderTest(unittest.TestCase):
    """Unit test for CHAP.common.BinaryFileReader class"""

    def setUp(self):
        self.reader = NexusReader(
            filename=os.path.join(test_data_dir, 'file.nxs'),
        )

    def testReader(self):
        """Unit test to test reader"""
        from nexusformat.nexus import NXroot
        self.reader.nxpath = '/'
        data = self.reader.read()
        self.assertIsInstance(data, NXroot)

    def testNXpath(self):
        """Unit test to test the `nxpath` keyword argument of
        `NexusReader.read`
        """
        from nexusformat.nexus import NXdata
        self.reader.nxpath = '/entry/data'
        data = self.reader.read()
        self.assertIsInstance(data, NXdata)


class URLReaderTest(unittest.TestCase):
    """Unit test for CHAP.common.URLReader class"""

    def setUp(self):
        self.reader = URLReader(filename='tbd', url='tbd')

    def testReader(self):
        """Unit test to test reader"""
        # data = self.reader.read(self.url)
        # self.assertIsInstance(data, bytes)


class YAMLReaderTest(unittest.TestCase):
    """Unit test for CHAP.common.YAMLReader class"""

    def setUp(self):
        self.reader = YAMLReader(
            filename=os.path.join(test_data_dir, 'file.yaml')
        )

    def testReader(self):
        """Unit test to test reader"""
        data = self.reader.read()
        self.assertIsInstance(data, dict)


if __name__ == '__main__':
    unittest.main()
