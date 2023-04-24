#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Processor module
"""

# system modules
from time import time

# local modules
from CHAP import Processor


class TFaaSImageProcessor(Processor):
    """A Processor to get predictions from TFaaS inference server."""

    def process(self, data, url, model, verbose=False):
        """process data API"""

        t0 = time()
        self.logger.info(f'Executing "process" with url {url} model {model}')

        data = self._process(data, url, model, verbose)

        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')

        return data

    def _process(self, data, url, model, verbose):
        """Print and return the input data.

        :param data: Input image data, either file name or actual
            image data
        :type data: object
        :return: `data`
        :rtype: object
        """
        # system modules
        from pathlib import Path

        # local modules
        from MLaaS.tfaas_client import predictImage

        self.logger.info(f'input data {type(data)}')
        if isinstance(data, str) and Path(data).is_file():
            img_file = data
            data = predictImage(url, img_file, model, verbose)
        else:
            # third party modules
            from requests import Session

            rdict = data[0]
            img = rdict['data']
            session = Session()
            rurl = url + '/predict/image'
            payload = {'model': model}
            files = {'image': img}
            self.logger.info(
                f'HTTP request {rurl} with image file and {payload} payload')
            req = session.post(rurl, files=files, data=payload)
            data = req.content
            data = data.decode('utf-8').replace('\n', '')
            self.logger.info(f'HTTP response {data}')

        return data


if __name__ == '__main__':
    # local modules
    from CHAP.processor import main

    main()
