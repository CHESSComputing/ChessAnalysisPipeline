#!/usr/bin/env python
"""
Standard python setup.py file
to build     : python setup.py build
to install   : python setup.py install --prefix=<some dir>
to clean     : python setup.py clean
to build doc : python setup.py doc
to run tests : python setup.py test
"""

import os
import setuptools

# [set version]
version = 'PACKAGE_VERSION'
# [version set]

def datafiles(idir, pattern=None):
    """Return list of data files in provided relative dir"""
    files = []
    for dirname, dirnames, filenames in os.walk(idir):
        for subdirname in dirnames:
            files.append(os.path.join(dirname, subdirname))
        for filename in filenames:
            if  filename[-1] == '~':
                continue
            # match file name pattern (e.g. *.css) if one given
            if pattern and not fnmatch.fnmatch(filename, pattern):
                continue
            files.append(os.path.join(dirname, filename))
    return files

data_files = datafiles('examples')

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ChessAnalysisPipeline",
    version=version,
    author="Keara Soloway, Rolf Verberg, Valentin Kuznetsov",
    author_email="",
    description="CHESS analysis pipeline framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CHESSComputing/ChessAnalysisPipeline",
    packages=[
        'CHAP',
        'CHAP.common',
        'CHAP.common.models',
        'CHAP.common.utils',
        'CHAP.edd',
        'CHAP.inference',
        'CHAP.saxswaxs',
        'CHAP.sin2psi',
        'CHAP.tomo',
        'MLaaS'
    ],
    package_dir={
        'CHAP': 'CHAP',
        'CHAP.common': 'CHAP/common',
        'CHAP.common.models': 'CHAP/common/models',
        'CHAP.common.utils': 'CHAP/common/utils',
        'CHAP.edd': 'CHAP/edd',
        'CHAP.inference': 'CHAP/inference',
        'CHAP.saxswaxs': 'CHAP/saxswaxs',
        'CHAP.sin2psi': 'CHAP/sin2psi',
        'CHAP.tomo': 'CHAP/tomo',
        'MLaaS': 'MLaaS'
    },
    package_data={
        'examples': data_files
    },
    entry_points={
        'console_scripts': ['CHAP = CHAP.runner:main']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'PyYAML'
    ],
)
