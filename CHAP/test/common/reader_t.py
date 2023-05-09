"""
File       : common/reader.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Unit tests for common/reader.py code
"""

# system modules
import time
import unittest

# local modules
from CHAP.common import YAMLReader


class CommonReaderTest(unittest.TestCase):
    """Unit test for common/reader.py module"""

    def setUp(self):
        self.fname = '/'.join(__file__.split("/")[:-2]) + '/data/file.yaml'
        self.reader = YAMLReader()

    def tierDown(self):
        pass

    def testReader(self):
        """
        Unit test to test reader
        """
        data = self.reader.read(self.fname)
        self.assertEqual(isinstance(data, dict), True)


if __name__ == '__main__':
    unittest.main()
