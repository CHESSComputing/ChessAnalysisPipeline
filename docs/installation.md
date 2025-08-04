# Installation


## Using `pip`
CHAP is available over PyPI and can be installed with:
```bash
pip install ChessAnalysisPipeline
```

## Using `conda`
CHAP is also available over conda-forge, allowing installation with, e.g.:
```bash
conda install -c conda-forge chessanalysispipeline
```

## From source
1. Clone the CHAP repository:
   ```bash
   git clone https://github.com/CHESSComputing/ChessAnalysisPipeline.git
   ```
1. Change to the CHAP repository directory:
   ```bash
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
1. Create the virtual environment:
   ```bash
   python -m venv venv
   ```
1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
1. Use the setup script to install the package:
   ```bash
   python setup.py install
   ```
1. Try running:
   ```bash
   install/bin/CHAP --help
   ```
   to confirm the package was installed correctly.

If you need to install it in your local installation area via pip, run
steps 1 to 3 from above and then run the following steps:

1. Install the wheel package:
   ```bash
   pip install wheel
   ```
1. Build your local package:
   ```bash
   python setup.py clean sdist bdist_wheel
   ```
1. Look-up your local package in dist area:
   ```bash
   ls -1 dist
   ```
   which should return:
   ```bash
   ChessAnalysisPipeline-0.0.16-py3-none-any.whl
   ChessAnalysisPipeline-0.0.16.tar.gz
   ```
1. Install your package from the local dist area:
   ```bash
   pip install --no-index --find-links=dist/ ChessAnalysisPipeline
   ```
1. Verify that you package is installed:
   ```bash
   pip list | grep Chess
   ```
   which should return:
   ```bash
   ChessAnalysisPipeline 0.0.16
   ```
1. Try running:
   ```bash
   install/bin/CHAP --help
   ```
   to confirm the package was installed correctly.

