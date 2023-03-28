#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : async.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: AsyncProcessor module
"""

# system modules
import asyncio

# local modules
from CHAP.processor import Processor, PrintProcessor


async def task(mgr, doc):
    """
    Process given data using provided task manager
    """
    return mgr.process(doc)


async def executeTasks(mgr, docs):
    """
    Process given set of documents using provided task manager
    """
    coRoutines = [task(mgr, d) for d in docs]
    await asyncio.gather(*coRoutines)


class AsyncProcessor(Processor):
    """
    AsyncProcesor process given data via asyncio module
    """
    def __init__(self, mgr):
        super().__init__()
        self.mgr = mgr

    def _process(self, docs):
        """
        Internal method to process given data documents
        """
        asyncio.run(executeTasks(self.mgr, docs))

def example():
    """
    Helper function to demonstrate usage of AsyncProcessor
    """
    docs = [1,2,3]
    mgr = PrintProcessor()
    processor = AsyncProcessor(mgr)
    processor.process(docs)

if __name__ == '__main__':
    example()
