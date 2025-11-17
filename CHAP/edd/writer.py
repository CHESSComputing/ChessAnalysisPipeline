#!/usr/bin/env python
"""EDD specific writers."""

# Local modules
from CHAP import Writer

class StrainAnalysisUpdateWriter(Writer):
    def write(self, data):
        # System modules
        from os import path as os_path

        # Third party modules
        from nexusformat.nexus import nxload

        # Local modules
        from CHAP.edd.processor import StrainAnalysisProcessor

        points = self.unwrap_pipelinedata(data)[0]
        nxroot = nxload(self.filename, mode='r+')
        StrainAnalysisProcessor.add_points(nxroot, points, logger=self.logger)

        return nxroot


if __name__ == '__main__':
    # Local modules
    from CHAP.writer import main

    main()
