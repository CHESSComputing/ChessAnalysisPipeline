# EDD subpackage (CHAP.edd)

The EDD subpackage contains the modules that are unique to Energy Dispersive Diffraction (EDD) data processing workflows. This document describes how to run an detector energy calibration and strain analysis workflow in a Linux terminal.

## Processing the data

A standard strain analysis in CHAP consists of three steps:

- Performing the detector channel energy calibration. This is typically performed by fitting a set of fluorescence peak centers in an EDD experiment on a CeO2 sample and comparing the results to their known energy values.

- Fine tuning the detector channel energy calibration (and optionally the takeoff angle $`2\theta`$) by fitting a set of Bragg peak centers in an EDD experiment on typically the same CeO2 sample and comparing the results to their known energy values for a given channel energy calibration (and $`2\theta`$ value).

- Performing the strain analysis with an EDD experiment on a sample using the calibrated detector channel energies.

## Activating the EDD conda environment

### From the CHESS Compute Farm

Log in to the CHESS Compute Farm and activate the `CHAP_edd` environment:
```bash
source /nfs/chess/sw/miniforge3_chap/bin/activate
```
```bash
conda activate CHAP_edd
```

### From a local CHAP clone

1. Create and activate a base conda environent, e.g. with [Miniforge](https://github.com/conda-forge/miniforge).
1. Install a local version of the CHAP package according to the [instructions](/docs/installation.md)
1. Create the EDD conda environment:
   ```bash
   mamba env create -f <path_to_CHAP_clone_dir>/CHAP/edd/environment.yml
   ```
1. Activate the `CHAP_edd` environment:
   ```bash
   conda activate CHAP_edd
   ```

## Running an EDD workflow

1. Navigate to your work directory.
1. Create the required CHAP pipeline file for the workflow (see below) and any additional workflow specific input files. 
1. Run the workflow:
   ```bash
   CHAP <pipelinefilename>
   ```
1. Respond to any prompts that pop up if running interactively.

## Inspecting output

The output consists of a single NeXus (`.nxs`) file containing the strain analysis data as well as all metadata pertaining to the analysis. Additionally, optional output figures (`.png`) may be saved to an output directory specified in the pipeline file.

The optional output figures can be viewed directly by any PNG image viewer. The data in the NeXus output file can be viewed in [NeXpy](https://nexpy.github.io/nexpy/), a high-level python interface to HDF5 files, particularly those stored as [NeXus data](http://www.nexusformat.org):

1. Open the NeXpy GUI by entering in your terminal:
   ```bash
   nexpy &
   ```
1. After the GUI pops up, click File-> Open to navigate to the folder where your output `.nxs` file was saved, and select it.
1. Navigate the filetree in the "NeXus Data" panel to inspect any output or metadata field.

## Creating the pipeline file

Create a workflow `pipeline.yaml` file according to the [instructions](/docs/pipeline.md). A generic pipeline input file for an energy calibration and strain analysis workflow is as follows (note that spaces and indentation are important in `.yaml` files):
```
config:
  root: .           # Change as desired
  inputdir: .       # Change as desired
                    # Path can be relative to root (line 2) or absolute
  outputdir: output # Change as desired
                    # Path can be relative to root (line 2) or absolute
  interactive: true # Change as desired
  log_level: info   # Set to debug, info, warning, or error

pipeline:

  # Energy calibration
  - common.SpecReader:
      config:
        station: id3a # Change as needed
        experiment_type: EDD
        spec_scans: # Edit both SPEC log file path and EDD scan numbers
                    # Path can be relative to inputdir (line 3) or absolute
          - spec_file: <your_raw_ceria_data_directory>/spec.log
            scan_numbers: 1
  - edd.MCAEnergyCalibrationProcessor:
      config:
        max_peak_index: 1
        peak_energies: [34.276, 34.717, 39.255, 40.231]
        materials:  # Use default CeO2 properties when omitted
          - material_name: CeO2
            sgnum: 225
            lattice_parameters: 5.41153
      detector_config:
        baseline: true
        mask_ranges: [[650, 850]]
        detectors:  # Choose the detectors
                    # Use all available detector elements when omitted
          - id: 0
          - id: 11
          - id: 22
      save_figures: true
      schema: edd.models.MCAEnergyCalibrationConfig

  # Twotheta calibration
  - common.SpecReader:
      config:
        station: id3a # Change as needed
        experiment_type: EDD
        spec_scans: # Edit both SPEC log file path and EDD scan numbers
                    # Path can be relative to inputdir (line 2) or absolute
          - spec_file: <your_raw_ceria_data_directory>/spec.log
            scan_numbers: 1
  - edd.MCATthCalibrationProcessor:
      config:
        tth_initial_guess: 6.0
      detector_config:
        energy_mask_ranges: [[81, 135]] # Change as needed
        detectors:  # The same as in the energy calibration when omitted
          - id: 0
          - id: 11
          - id: 22
      save_figures: true

  # Create a CHESS style map
  - edd.EddMapReader:
      filename: <your_raw_data_directory>/id1a3-wbmapscan-s23-map-1.par
      dataset_id: 1
      schema: common.models.map.MapConfig
  - common.MapProcessor:
      detector_config:
        detectors:  # Use available detector elements when omitted
          - id: 0
          - id: 11
          - id: 22

  # Perform the strain analysis
  - pipeline.MultiplePipelineItem:
      items:
        - common.YAMLReader:
            filename: tth_calibration_result.yaml
            schema: edd.models.MCATthCalibrationConfig
  - edd.StrainAnalysisProcessor:
      config:
        materials:
          - material_name: Al
            sgnum: 225
            lattice_parameters: 4.046
        rel_height_cutoff: 0.05         # Change as desired
        skip_animation: false
      detector_config:
        energy_mask_ranges: [[77, 110]] # Change as desired
        detectors: # # Use available detector elements when omitted
          - id: 0
          - id: 11
          - id: 22
      save_figures: true
  - common.NexusWriter:
      filename: strain_map.nxs # Change as desired
                               # will be placed in 'outdutdir' (line 3)
      force_overwrite: true # Do not set to false!
                            # Rename an existing file if you want to prevent
                            # it from being overwritten
```

The "config" block defines the CHAP generic configuration parameters:

- `root`: The work directory, defaults to the current directory (where `CHAP <pipelinefilename>` is executed). Must be an absolute path or relative to the current directory.

- `inputdir`: The default directory for files read by any CHAP reader (must have read access), defaults to `root`. Must be an absolute path or relative to `root`.

- `outputdir`: The default directory for files written by any CHAP writter (must have write access, will be created if not existing), defaults to `root`. Must be an absolute path or relative to `root`.

- `interactive`: Allows for user interactions, defaults to `False`.

- `log_level`: The [Python logging level](https://docs.python.org/3/library/logging.html#levels).

The "pipeline" block creates the actual workflow pipeline, in this example it consists of nine toplevel processes that get executed successively:

- The EDD/XRF energy calibration consists of two processes:

    - `common.SpecReader`: A processor that reads the raw detector data.

    - `edd.MCAEnergyCalibrationProcessor`: A processor that performs the detector channel energy calibration.

- The EDD $`2\theta`$ calibration consists of two processes:

    - `common.SpecReader` under `pipeline.MultiplePipelineItem`: A processor that reads the raw detector data and adds it to the energy calibration info that gets passed along directly from the previous processor in the pipeline.

    - `edd.MCATthCalibrationProcessor`: A processor that performs the $`2\theta`$ calibration.

- The following two processors read the strain analysis sample's raw detector data and create a CHESS style map:

    - `edd.EddMapReader`: A processor that reads the sample's raw detector data using a CHESS style experiment par-file.

    - `common.MapProcessor`: A processor that creates a CHESS style map for the raw detector data.

- The last three processors perform the actual strain analysis and write the output to a Nexus file:

    - `common.YAMLReader` under `pipeline.MultiplePipelineItem`: A processor that reads the energy/$`2\theta`$ calibration results adds it to the raw data map that gets passed along directly from the previous processor in the pipeline.

    - `edd.StrainAnalysisProcessor`: A processor that perfroms the actual strain analysis and creates a single Nexus object with the strain analysis results as well as all metadata pertaining to the workflow.

    - `common.NexusWriter`: A processor that writes the strain analysis results to a NeXus file.

Note that the energy calibration can also be obtained ahead of time and used for multiple strain analyses. In this case remove the first four processes in the pipeline and read the detector channel energy calibration info in what is now the third item in the pipeline.

## Additional notes on energy calibration

As mentioned above a standard EDD experiment needs calibration of the detector channel energies. Experiments have shown that the channel energies $`E_j`$ vary linearly with the channel index $`j`$ within the energy range of typical EDD experiments: $`E_j = mj+b`$, where the slope $`m`$ and intercept $`b`$ can be determined in one or a combination of two experiments:

1. With an XRF experiment by fitting a set of flueorescence peak centers at known energies:

This uniquely determines $`m`$ and $`b`$ within the statistical errors of the experiment without having to know the actual takeoff angle $`2\theta`$.

1. With a diffraction experiment by fitting a set of Bragg peak centers corresponding to known lattice spacings $`d_{hkl}`$:

Given Bragg's law, $`\lambda = 2d\sin(\theta)`$, with $`E = hc/\lambda`$, the Bragg peaks appear at channels with energies $`E_{hkl} = hc / (2d_{hkl}\sin(\theta)`$. Rearranging this with the detector channel energy calibration relation gives:
```math
\frac{1}{d_{hkl}} = \frac{2m\sin(\theta)}{hc} j_{hkl} + \frac{2b\sin(\theta)}{hc} = m'j_{hkl}+b'
```
which says that given a set of known Bragg peaks corresponding to lattice spacings $`d_{hkl}`$ occuring at channel indices $`j_{hkl}`$, a linear fit will uniquely determine $`m'`$ and $`b'`$. For a known takeoff angle $`2\theta`$, this uniquely determines $`m`$ and $`b`$ as well. Note that this also implies that without an accurately known value of $2`theta`$, one *cannot* uniquely determine $`m`$ and $`b`$ from Bragg diffraction alone!

This leads to the above mentioned two-step detector channel energy calibration procedure:

1. Get nominal values for $`m`$ and $`b`$, by performing an EDD/XRF experiment on a Ce02 sample and fitting a set of fluorescence peak centers with known energies vs detector channel index.

1. Fine tuning the calibration by fitting a set of Bragg peak centers in an EDD experiment on (typically) the same CeO2 sample, where the channel indices for the initial Bragg peak positions are obtained from the known Bragg peak energies and the nominal values for $`m`$ and $`b`$:

    - If $`2\theta`$ is known with sufficient accuracy, one can fit the peak centers vs detector channel index to get $`m'`$ and $`b'`$ and thus with the known $`2\theta`$ convert those directly to fine tuned values for $`m`$ and $`b`$ over the entire energy range of interest.

    - If $`2\theta`$ is not known with sufficient accuracy, one can fit the Bragg peak centers vs detector channel index to get $`m`$, $`b`$ and $`2\theta`$ in a way that minimizes the RMS error between the fitted peak centers for a given fit parameter set $`(m, b, 2\theta)`$ and those obtained from Bragg's law given their $`d_{hkl}`$'s. The latter will still need a sufficently decent initial guess for $`2\theta`$, which can be given as an input to the CHAP $`2\theta`$ calibration processor or picked interactively.


The choice between the latter two apporaches is set by the `calibration_method` field in the $`2\theta`$ processor configuration in the pipeline yaml input file. Set `calibration_method` to `direct_fit_bragg` (default) to use a fixed given value of $`2\theta`$, or set it to `direct_fit_tth_ecc` to fit for the unknown $`2\theta`$ and the energy calibration coefficients $`m`$ and $`b`$.





