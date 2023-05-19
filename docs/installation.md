# Installation


## Using `pip`
CHAP is available over PyPI. Run
```
pip install ChessAnalysisPipeline
```
to install.

## Using `conda`
CHAP is available over conda-forge. Run
```
conda install chessanalysispipeline
```
to install.

## From source
1. Clone the CHAP repository.
   ```{bash}
   git clone https://github.com/CHESSComputing/ChessAnalysisPipeline.git
   cd ChessAnalysisPipeline
   ```
1. Set a valid version number. In `setup.py`, replace:
   ```{literalinclude} /../setup.py
   :start-after: [set version]
   :end-before: [version set]
   ```
   with
   ```
   version = 'PACKAGE_VERSION'
   ```
1. Setup a local installation prefix for your version of python
   ```{bash}
   mkdir -p install/lib/python<yourpythonversion>/site-packages
   export PYTHONPATH=$PWD/install/lib/python<yourpythonversion>/site-packages
   ```
1. Use the setup script to install the package
   ```
   python setup.py install --prefix=$PWD/install
   ```
1. Try running
   ```
   install/bin/CHAP --help
   ```
   to confirm the package was installed correctly.