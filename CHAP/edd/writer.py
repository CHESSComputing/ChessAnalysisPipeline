#!/usr/bin/env python
"""EDD specific writers."""

# Local modules
from CHAP import Writer

class StrainAnalysisUpdateWriter(Writer):
    def write(self, data, filename, force_overwrite=True):
        # Third party modules
        from nexusformat.nexus import nxload

        # Local modules
        from CHAP.edd.processor import StrainAnalysisProcessor

        points = self.unwrap_pipelinedata(data)[0]
        nxroot = nxload(filename, mode='r+')
        StrainAnalysisProcessor.add_points(nxroot, points)

        return nxroot


if __name__ == '__main__':
    # Local modules
    from CHAP.writer import main

    main()
