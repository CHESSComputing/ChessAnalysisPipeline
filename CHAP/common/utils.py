#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File       : utils.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: common set of utility functions
"""

# System modules
import os
import sys
import platform
import pkg_resources
import subprocess
import json

def osinfo():
    """Helper function to provide osinfo."""
    os_info = {
        'name': f'{platform.system().lower()}-{platform.release()}',
        'kernel': platform.version(),
        'version': platform.platform(),
    }
    return os_info

def environments():
    """Detects the current Python environment (system, virtualenv,
    Conda, or pip) and collects package information. Returns a list
    of detected environments with installed packages.
    """
    environments = []
    os_name = f'{platform.system().lower()}-{platform.release()}'

    # Check for Conda environment
    conda_env = os.getenv('CONDA_PREFIX')
    if conda_env:
        conda_env_name = os.getenv('CONDA_DEFAULT_ENV', 'unknown-conda-env')
        try:
            # Fetch Conda packages
            conda_packages = subprocess.check_output(
                ['conda', 'list', '--json'], text=True)
            conda_packages = json.loads(conda_packages)
            packages = [{'name': pkg['name'], 'version': pkg['version']}
                        for pkg in conda_packages]
        except Exception:
            packages = []
        environments.append({
            'name': conda_env_name,
            'version': sys.version.split()[0],
            'details': 'Conda environment',
            'parent_environment': None,
            'os_name': os_name,
            'packages': packages,
        })

    # Check for Virtualenv (excluding Conda)
    elif hasattr(sys, 'real_prefix') or os.getenv('VIRTUAL_ENV'):
        venv_name = os.path.basename(os.getenv('VIRTUAL_ENV', 'unknown-venv'))
        packages = [{'name': pkg.key, 'version': pkg.version}
                    for pkg in pkg_resources.working_set]
        environments.append({
            'name': venv_name,
            'version': sys.version.split()[0],
            'details': 'Virtualenv environment',
            'parent_environment': None,
            'os_name': os_name,
            'packages': packages,
        })

    # System Python (not inside Conda or Virtualenv)
    else:
        packages = [{'name': pkg.key, 'version': pkg.version}
                    for pkg in pkg_resources.working_set]
        environments.append({
            'name': 'system-python',
            'version': sys.version.split()[0],
            'details': 'System-wide Python',
            'parent_environment': None,
            'os_name': os_name,
            'packages': packages,
        })

    # Check for PIP installed editable packages
    try:
        # Fetch Git repo info for PIP installed editable packages
        pip_packages = subprocess.check_output(
            ['pip', 'list', '--format', 'json'], text=True)
        pip_packages = json.loads(pip_packages)
        packages = [{'name': pkg['name'],
                     #'version': pkg['version'],
                     'commit_hash': subprocess.check_output(
                         ["git", "rev-parse", "HEAD"],
                         cwd=pkg['editable_project_location']
                     ).strip().decode()}
                    for pkg in pip_packages
                    if 'editable_project_location' in pkg]
        environments.append({
            'name': 'Python Package Installer',
            'version': sys.version.split()[0],
            'details': 'Editable project locations',
            'parent_environment': None,
            'os_name': os_name,
            'packages': packages,
        })
    except Exception:
        pass

    return environments
