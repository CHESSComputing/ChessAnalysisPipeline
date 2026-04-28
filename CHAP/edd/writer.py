#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""Module for Writers unique to the EDD workflow."""

# Local modules
from CHAP.writer import Writer

class StrainAnalysisUpdateWriter(Writer):
    """Writer to add or update the strain analysis for a set of map
    points.
    """

    def write(self, data):
        """Write or update strain analysis results for a set of points.

        :param data: Input data.
        :type data: list[PipelineData]
        """
        # Third party modules
        from nexusformat.nexus import nxload

        # Local modules
        from CHAP.edd.processor import StrainAnalysisProcessor

        points = self.get_pipelinedata_item(data, remove=self.remove)
        nxroot = nxload(self.filename, mode='r+')
        StrainAnalysisProcessor.add_points(nxroot, points, logger=self.logger)


if __name__ == '__main__':
    # Local modules
    from CHAP.writer import main

    main()
