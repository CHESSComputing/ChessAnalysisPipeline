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
1. If you need to install it in your local installation area via pip please
   follow the following steps:
```
# install wheel package
pip install wheel

# build your local package
python setup.py clean sdist bdist_wheel

# look-up your local package in dist area
ls -1 dist
ChessAnalysisPipeline-0.0.16-py3-none-any.whl
ChessAnalysisPipeline-0.0.16-py3.11.egg
ChessAnalysisPipeline-0.0.16.tar.gz

# install your package from local dist area
pip install --no-index --find-links=dist/ ChessAnalysisPipeline

# verify that you package is installed
pip list | grep Chess
ChessAnalysisPipeline 0.0.16
```
