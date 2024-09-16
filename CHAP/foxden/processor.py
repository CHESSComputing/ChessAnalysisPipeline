#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Processor module for FOXDEN services
"""

# System modules
from time import time

# Local modules
from CHAP.processor import Processor

class FoxdenProvenanceProcessor(Processor):
    """A Processor to communicate with FOXDEN provenance server."""
#    def __init__(self):
#        self.writer = FoxdenWriter()

    def process(self, data, url, dryRun=False, verbose=False):
        """Process the data API."""
        # Local modules
        from CHAP.foxden.writer import FoxdenWriter

        t0 = time()
        self.logger.info(
            f'Executing "process" with url {url} data {data} dryrun {dryRun}')
        writer = FoxdenWriter()

#        data = self.writer.write(data, url, dryRun)
        data = writer.write(data, url, dryRun=dryRun)

        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')

        return data


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
