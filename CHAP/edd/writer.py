#!/usr/bin/env python
"""EDD specific writers."""

# Local modules
from CHAP import Writer

class StrainAnalysisUpdateWriter(Writer):
    def write(self, data, filename, force_overwrite=True):
        # System modules
        from os import path as os_path

        # Third party modules
        from nexusformat.nexus import nxload

        # Local modules
        from CHAP.edd.processor import StrainAnalysisProcessor

        if os_path.isfile(filename) and not force_overwrite:
            raise FileExistsError(f'{filename} already exists')

        points = self.unwrap_pipelinedata(data)[0]
        nxroot = nxload(filename, mode='r+')
        StrainAnalysisProcessor.add_points(nxroot, points, logger=self.logger)

        return nxroot


if __name__ == '__main__':
    # Local modules
    from CHAP.writer import main

    main()
