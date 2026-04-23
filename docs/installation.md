(installation)=
# Installation


## Using `pip`
``CHAP`` is available over PyPI and can be installed with:
```bash
$ pip install ChessAnalysisPipeline
```

## Using `conda`
`CHAP` is also available over conda-forge, allowing installation with, e.g.:
```bash
$ conda install -c conda-forge chessanalysispipeline
```

## Installing CHAP from source on a Linux system

(virtual_env_installation)=
### Using a Python virtual environment

1. Clone the `CHAP` repository:
   ```bash
   $ git clone https://github.com/CHESSComputing/ChessAnalysisPipeline.git
   ```
1. Change to the `CHAP` repository directory:
   ```bash
   $ cd ChessAnalysisPipeline
   ```
1. Set a valid version number. In `setup.py`, replace:
   ```{literalinclude} /../setup.py
   :language: python
   :start-after: '[set version]'
   :end-before: '[version set]'
   ```
   with (pick the appropriate version string):
   ```python
   version = 'v1.1'
   ```
1. Make sure that your Python version is at least 3.10. To check your Python version, execute:
   ```bash
   $ python --version
   ```
1. Create the [Python virtual environment](https://docs.python.org/3/library/venv.html):
   ```bash
   $ python -m venv venv
   ```
   or use, for example:
   ```bash
   $ python3.10 -m venv venv
   ```
   if you have multiple Python versions on your system including python3.10 and your default version is below v3.10.
1. Activate the virtual environment:
   ```bash
   $ source venv/bin/activate
   ```
1. Use the setup script to install the package:
   ```bash
   $ python setup.py install
   ```
   You may have to install `setuptools` if this returns
   ```
   ModuleNotFoundError: No module named 'setuptools'
   ```
   In this case, run:
   ```bash
   $ pip install setuptools
   ```
1. Install the required Python packages:
   ```bash
   $ pip install -r requirements.txt
   ```
1. Try running:
   ```bash
   $ CHAP --help
   ```
   to confirm the package was installed correctly.
1. Note that you have to reinstall the package again when you want code changes to have an effect. To avoid this you can create an editable `CHAP` installation:
   ```bash
   $ pip install -e .
   ```

(conda_installation)=
### Using a Conda environment

1. Create a base Conda environent, e.g. with [Miniforge](https://github.com/conda-forge/miniforge). Download the installer appropriate for your computer's architecture using curl or wget or your favorite program, for example:
   ```bash
   $ wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   ```
   Run the script with:
   ```bash
   $ bash Miniforge3-$(uname)-$(uname -m).sh
   ```
   Then accept the license terms, pick the default or choose a suitable Miniforge3 installation directory, and select "no" or press enter when prompted to proceed with the initialization (unless you want the script to make changes to you shell profile to automatically initialize conda, which is not recommended).
1. Navigate to a suitable directory and clone the `CHAP` repository:
   ```bash
   $ git clone https://github.com/CHESSComputing/ChessAnalysisPipeline.git
   ```
1. Change to the `CHAP` repository directory:
   ```bash
   $ cd ChessAnalysisPipeline
   ```
1. Activate the miniforge3 base environment:
   ```bash
   $ source <path_to_miniforge3_dir>/bin/activate
   ```
1. Create the `CHAP` environment:
   ```bash
   (bash) $ mamba env create -f environment.yml
   ```
1. Activate the `CHAP` environment:
   ```bash
   (bash) $ conda activate CHAP
1. Set a valid version number. In `setup.py`, replace:
   ```{literalinclude} /../setup.py
   :language: python
   :start-after: '[set version]'
   :end-before: '[version set]'
   ```
   with (pick the appropriate version string):
   ```python
   version = 'v1.1'
   ```
1. Create an editable `CHAP` installation:
   ```bash
   (CHAP) $ pip install -e .
   ```
   or use the setup script to install the package:
   ```bash
   (CHAP) $ python setup.py install
   ```
1. Try running:
   ```bash
   (CHAP) $ CHAP --help
   ```
   to confirm the package was installed correctly.
