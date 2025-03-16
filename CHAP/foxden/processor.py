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
import os
import sys
import platform
import pkg_resources
import subprocess
import json

# Local modules
from CHAP.processor import Processor

class FoxdenMetaDataProcessor(Processor):
    """A Processor to communicate with FOXDEN MetaData server."""

    def process(self, data, suffix='analysis=CHAP', verbose=False):
        """FOXDEN MetaData processor

        :param data: Input data.
        :type data: list[PipelineData]
        :param suffix: did suffix to add, default 'analysis=CHAP'
        :type suffix: string, optional
        :param verbose: verbose output
        :type verbose: bool, optional
        :return: data from FOXDEN MetaData service
        """
        t0 = time()
        self.logger.info(
            f'Executing "process" with data={data}')
        output = []
        for item in data:
            # each item in data list is a CHAP record {'name': ..., 'data': {}}
            for rec in item['data']:  # get data part of processing item
                if 'did' not in rec:
                    raise Exception('No did found in input data record')
                did = rec['did'] + '/' + suffix
                # construct analysis record
                rec = {'did': did, 'application': 'CHAP'}
                output.append(rec)
        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')
        return output

class FoxdenProvenanceProcessor(Processor):
    """A Processor to communicate with FOXDEN provenance server."""
    def process(self, data, suffix='analysis=CHAP', verbose=False):
        """FOXDEN Provenance processor

        :param data: Input data.
        :type data: list[PipelineData]
        :param suffix: did suffix to add, default 'analysis=CHAP'
        :type suffix: string, optional
        :param verbose: verbose output
        :type verbose: bool, optional
        :return: data from FOXDEN provenance service
        """
        t0 = time()
        self.logger.info(
            f'Executing "process" with data={data}')
        output = []
        for item in data:
            # each item in data list is a CHAP record {'name': ..., 'data': {}}
            for rec in item['data']:  # get data part of processing item
                if 'did' not in rec:
                    raise Exception('No did found in input data record')
                rec['did'] = rec['did'] + '/' + suffix
                rec['parent_did'] = rec['did'] 
                rec['scripts'] = [{'name': 'CHAP', 'parent_script': None, 'order_idx': 1}]
                rec['site'] = 'Cornell'
                rec['osinfo'] = osinfo()
                rec['environments'] = environments()
                rec['input_files'] = inputFiles()
                rec['output_files'] = outputFiles()
                rec['processing'] = 'CHAP pipeline'
                output.append(rec)
        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')
        return output

def osinfo():
    """
    Helper function to provide osinfo
    """
    os_info = {
        "name": platform.system().lower() + "-" + platform.release(),
        "kernel": platform.version(),
        "version": platform.platform()
    }
    return os_info

def environments():
    """
    Detects the current Python environment (system, virtualenv, or Conda) and
    collects package information. Returns a list of detected environments with
    installed packages.
    """
    environments = []
    os_name = platform.system().lower() + "-" + platform.release()

    # Check for Conda environment
    conda_env = os.getenv("CONDA_PREFIX")
    if conda_env:
        conda_env_name = os.getenv("CONDA_DEFAULT_ENV", "unknown-conda-env")
        try:
            # Fetch Conda packages
            conda_packages = subprocess.check_output(["conda", "list", "--json"], text=True)
            conda_packages = json.loads(conda_packages)
            packages = [{"name": pkg["name"], "version": pkg["version"]} for pkg in conda_packages]
        except Exception:
            packages = []

        environments.append({
            "name": conda_env_name,
            "version": sys.version.split()[0],
            "details": "Conda environment",
            "parent_environment": None,
            "os_name": os_name,
            "packages": packages
        })

    # Check for Virtualenv (excluding Conda)
    elif hasattr(sys, 'real_prefix') or os.getenv("VIRTUAL_ENV"):
        venv_name = os.path.basename(os.getenv("VIRTUAL_ENV", "unknown-venv"))
        packages = [
            {"name": pkg.key, "version": pkg.version}
            for pkg in pkg_resources.working_set
        ]

        environments.append({
            "name": venv_name,
            "version": sys.version.split()[0],
            "details": "Virtualenv environment",
            "parent_environment": None,
            "os_name": os_name,
            "packages": packages
        })

    # System Python (not inside Conda or Virtualenv)
    else:
        packages = [
            {"name": pkg.key, "version": pkg.version}
            for pkg in pkg_resources.working_set
        ]

        environments.append({
            "name": "system-python",
            "version": sys.version.split()[0],
            "details": "System-wide Python",
            "parent_environment": None,
            "os_name": os_name,
            "packages": packages
        })

    return environments

def inputFiles():
    """
    Helper function to provide input files for FOXDEN
    """
    return [{'name':'/tmp/file1.png'}, {'name': '/tmp/file2.png'}]

def outputFiles():
    """
    Helper function to provide output files for FOXDEN
    """
    return [{'name':'/tmp/file1.png'}]

if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
