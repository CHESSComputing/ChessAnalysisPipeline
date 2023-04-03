#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Processor module
"""

# system modules
import argparse
import json
import logging
import sys
from time import time

# local modules
# from pipeline import PipelineObject

class Processor():
    """
    Processor represent generic processor
    """
    def __init__(self):
        """
        Processor constructor
        """
        self.__name__ = self.__class__.__name__
        self.logger = logging.getLogger(self.__name__)
        self.logger.propagate = False

    def process(self, data):
        """
        process data API
        """

        t0 = time()
        self.logger.info(f'Executing "process" with type(data)={type(data)}')

        data = self._process(data)

        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')

        return(data)

    def _process(self, data):
        # If needed, extract data from a returned value of Reader.read
        if isinstance(data, list):
            if all([isinstance(d,dict) for d in data]):
                data = data[0]['data']
        # process operation is a simple print function
        data += "process part\n"
        # and we return data back to pipeline
        return data


class TFaaSImageProcessor(Processor):
    '''
    A Processor to get predictions from TFaaS inference server.
    '''
    def process(self, data, url, model, verbose=False):
        """
        process data API
        """

        t0 = time()
        self.logger.info(f'Executing "process" with url {url} model {model}')

        data = self._process(data, url, model, verbose)

        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')

        return(data)

    def _process(self, data, url, model, verbose):
        '''Print and return the input data.

        :param data: Input image data, either file name or actual image data
        :type data: object
        :return: `data`
        :rtype: object
        '''
        from MLaaS.tfaas_client import predictImage
        from pathlib import Path
        self.logger.info(f"input data {type(data)}")
        if isinstance(data, str) and Path(data).is_file():
            imgFile = data
            data = predictImage(url, imgFile, model, verbose)
        else:
            rdict = data[0]
            import requests
            img = rdict['data']
            session = requests.Session()
            rurl = url + '/predict/image'
            payload = dict(model=model)
            files = dict(image=img)
            self.logger.info(f"HTTP request {rurl} with image file and {payload} payload")
            req = session.post(rurl, files=files, data=payload )
            data = req.content
            data = data.decode("utf-8").replace('\n', '')
            self.logger.info(f"HTTP response {data}")

        return(data)

class OptionParser():
    '''User based option parser'''
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog='PROG')
        self.parser.add_argument(
            '--data', action='store',
            dest='data', default='', help='Input data')
        self.parser.add_argument(
            '--processor', action='store',
            dest='processor', default='Processor', help='Processor class name')
        self.parser.add_argument(
            '--log-level', choices=logging._nameToLevel.keys(),
            dest='log_level', default='INFO', help='logging level')

def main(opt_parser=OptionParser):
    '''Main function'''

    optmgr  = opt_parser()
    opts = optmgr.parser.parse_args()
    clsName = opts.processor
    try:
        processorCls = getattr(sys.modules[__name__],clsName)
    except:
        print(f'Unsupported processor {clsName}')
        sys.exit(1)

    processor = processorCls()
    processor.logger.setLevel(getattr(logging, opts.log_level))
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter('{name:20}: {message}', style='{'))
    processor.logger.addHandler(log_handler)
    data = processor.process(opts.data)

    print(f"Processor {processor} operates on data {data}")

if __name__ == '__main__':
    main()
