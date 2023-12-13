# Tomography subpackage (CHAP.tomo)

The tomography subpackage contains the modules that are unique to tomography data processing workflows. This document describes how to run a tomography reconstruction workflow in a Linux terminal.

A standard tomographic reconstruction in CHAP consists of three steps:

- Reducing the data, i.e., correcting the raw detector images for background and non-uniformities in the beam intensity profile using dark and bright fields collected separately from the tomography image series.

- Finding the calibrated rotation axis. Accurate reconstruction relies on accurately knowing the center of rotation at each data plane perpendicular to the rotation axis (the sinogram). This rotation axis is calibrated by selecting two data planes, one near the top and one near the bottom of the sample or beam, and visually or automatically picking the optimal center location.

- Reconstructing the reduced data for the calibrated rotation axis. For samples taller than the height of the beam, this last step can consist of two parts:

  - reconstruction of each individual stack of images, and

  - combining the individual stacks into one 3D reconstructed data set.

## Activating the tomography conda environment

### From the CHESS Compute Farm

Log in to the CHESS Compute Farm and activate the `CHAP_tomo` environment:
```bash
source /nfs/chess/sw/miniconda3_msnc/bin/activate
conda activate CHAP_tomo
```

### From a local CHAP clone

1. Create and activate a base conda environent, e.g. with [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/).
1. Install a local version of the CHAP package according to the [instructions](/docs/installation.md)
1. Create the tomography conda environment:
   ```bash
   conda env create -f <path_to_CHAP_clone_dir>/CHAP/tomo/environment.yml
   ```
1. Activate the `CHAP_tomo` environment:
   ```bash
   conda activate CHAP_tomo
   ```

## Running a tomography reconstruction

1. Navigate to your work directory.
1. Create the required CHAP pipeline file for the workflow and any additional workflow specific input files. FIX: Needs general instruction on pipeline files and specific instruction for tomo.
1. Run the reconstruction:
   ```bash
   CHAP <pipelinefilename>
   ```
1. Respond to any prompts that pop up if running interactively.

## Inspecting output

The output consists of a single NeXus (`.nxs`) file containing the reconstructed data set as well as all metadata pertaining to the reconstruction. Additionally, optional output figures (`.png`) may be save to an output directory specified in the pipeline file.

Any of the optional output figures can be viewed directly by any PNG image viewer. The data in the NeXus output file can be viewed in [NeXpy](https://nexpy.github.io/nexpy/), a high-level python interface to HDF5 files, particularly those stored as [NeXus data](http://www.nexusformat.org):
1. Open the NeXpy program in your terminal:
   ```bash
   nexpy &
   ```
1. In nexpy click File-> Open to navigate to the folder where your output `.nxs` file was saved, and select it.
1. Double click on the base level `NXroot` field in the leftmost "NeXus Data" panel to view the reconstruction. Note that the `NXroot` name is always the basename of the output file.
1. Or navigate the filetree in the "NeXus Data" panel to inspect any other output or metadata field. Note that the latest data set in any tomography reconstruction workflow is always available under the "data" `NXdata` field among the default `NXentry`'s fields (it is this data set that is opened in the viewer panel when double clicking the `NXroot` field). The default `NXentry` name is always the "title" field in the workflow's map configuration.

## Creating the pipeline input file
