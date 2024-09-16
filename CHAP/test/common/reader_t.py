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
from CHAP.common import (
    BinaryFileReader,
    NexusReader,
    URLReader,
    YAMLReader,
)


class BinaryFileReaderTest(unittest.TestCase):
    """Unit test for CHAP.common.BinaryFileReader class"""

    def setUp(self):
        self.reader = BinaryFileReader()
        self.filename = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data/img.png')

    def testReader(self):
        """Unit test to test reader"""
        data = self.reader.read(self.filename)
        self.assertIsInstance(data, bytes)


class NexusReaderTest(unittest.TestCase):
    """Unit test for CHAP.common.BinaryFileReader class"""

    def setUp(self):
        self.reader = NexusReader()
        self.filename = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data/file.nxs')
        self.nxpath = 'entry/data'

    def testReader(self):
        """Unit test to test reader"""
        from nexusformat.nexus import NXroot
        data = self.reader.read(self.filename)
        self.assertIsInstance(data, NXroot)

    def testNXpath(self):
        """Unit test to test the `nxpath` keyword argument of
        `NexusReader.read`
        """
        from nexusformat.nexus import NXdata
        data = self.reader.read(self.filename, nxpath=self.nxpath)
        self.assertIsInstance(data, NXdata)


class URLReaderTest(unittest.TestCase):
    """Unit test for CHAP.common.URLReader class"""

    def setUp(self):
        self.reader = URLReader()
        self.url = 'tbd'

    def testReader(self):
        """Unit test to test reader"""
        pass
        # data = self.reader.read(self.url)
        # self.assertIsInstance(data, bytes)


class YAMLReaderTest(unittest.TestCase):
    """Unit test for CHAP.common.YAMLReader class"""

    def setUp(self):
        self.reader = YAMLReader()
        self.filename = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data/file.yaml')

    def testReader(self):
        """Unit test to test reader"""
        data = self.reader.read(self.filename)
        self.assertIsInstance(data, dict)


if __name__ == '__main__':
    unittest.main()
