# Installation


## Using `pip`
CHAP is available over PyPI. Run
```bash
pip install ChessAnalysisPipeline
```
to install.

## Using `conda`
CHAP is available over conda-forge. Run
```bash
conda install chessanalysispipeline
```
to install.

## From source
1. Clone the CHAP repository.
   ```bash
   git clone https://github.com/CHESSComputing/ChessAnalysisPipeline.git
   cd ChessAnalysisPipeline
   ```
1. Set a valid version number. In `setup.py`, replace:
   ```{literalinclude} /../setup.py
   :language: python
   :start-after: '[set version]'
   :end-before: '[version set]'
   ```
   with
   ```python
   version = 'PACKAGE_VERSION'
   ```
1. Setup a local installation prefix for your version of python
   ```bash
   mkdir -p install/lib/python<yourpythonversion>/site-packages
   export PYTHONPATH=$PWD/install/lib/python<yourpythonversion>/site-packages
   ```
1. Use the setup script to install the package
   ```bash
   python setup.py install --prefix=$PWD/install
   ```
1. Try running
   ```bash
   install/bin/CHAP --help
   ```
   to confirm the package was installed correctly.