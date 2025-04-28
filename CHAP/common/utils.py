#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
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
