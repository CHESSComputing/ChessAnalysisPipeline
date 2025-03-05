# EDD subpackage (CHAP.edd)

The EDD subpackage contains the modules that are unique to Energy Dispersive Diffraction (EDD) data processing workflows. This document describes how to run an detector energy calibration and strain analysis workflow in a Linux terminal.

A standard strain analysis in CHAP consists of three steps:

- Performing the detector energy channel calibration. This is typically performed by fitting a set of fluorescence peak centers in an EDD experiment on a CeO2 sample and comparing the results to their known energy values.

1. Fine tuning the detector energy channel calibration (and optionally the takeoff angle 2`$\Theta$`) by fitting a set of Bragg peak centers in an EDD experiment on typically the same CeO2 sample and comparing the results to their known energy values for a given energy channel calibration (and 2`$\Theta$` value).

1. Performing the strain analysis on a sample using the above calibrated detector channel energies.

## Activating the EDD conda environment

### From the CHESS Compute Farm

Log in to the CHESS Compute Farm and activate the `CHAP_edd` environment:
```bash
source /nfs/chess/sw/miniforge3_chap/bin/activate
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

## Running an EDD reconstruction

1. Navigate to your work directory.
1. Create the required CHAP pipeline file for the workflow (see below) and any additional workflow specific input files. 
1. Run the reconstruction:
   ```bash
   CHAP <pipelinefilename>
   ```
1. Respond to any prompts that pop up if running interactively.

## Inspecting output

The output consists of a single NeXus (`.nxs`) file containing the strain analysis data as well as all metadata pertaining to the analysis. Additionally, optional output figures (`.png`) may be saved to an output directory specified in the pipeline file.

Any of the optional output figures can be viewed directly by any PNG image viewer. The data in the NeXus output file can be viewed in [NeXpy](https://nexpy.github.io/nexpy/), a high-level python interface to HDF5 files, particularly those stored as [NeXus data](http://www.nexusformat.org):

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
  inputdir: .       # Change as desired
  outputdir: output # Change as desired
  interactive: true # Change as desired
  log_level: INFO
  profile: false

pipeline:

  # Energy calibration
  - common.SpecReader:
      config:
        station: id3a
        experiment_type: EDD
        spec_scans: # Edit: spec.log path and tomography scan numbers
                    # Path can be relative to inputdir (line 2) or absolute
          - spec_file: <your_raw_ceria_data_directory>/spec.log
            scan_numbers: 1
  - edd.MCAEnergyCalibrationProcessor:
      config:
        baseline: true
        mask_ranges: [[650, 850]]
        max_peak_index: 1
        peak_energies: [34.276, 34.717, 39.255, 40.231]
        detectors:  # Use available detector elements when omitted
          - id: 0
          - id: 11
          - id: 22
        materials:  # Use default CeO2 properties when omitted
          - material_name: CeO2
            sgnum: 225
            lattice_parameters: 5.41153
      save_figures: true
      schema: edd.models.MCAEnergyCalibrationConfig

  # Twotheta calibration
  - pipeline.MultiplePipelineItem:
      items:
        - common.SpecReader:
            config:
              station: id3a
              experiment_type: EDD
              spec_scans: # Edit: spec.log path and tomography scan numbers
                          # Path can be relative to inputdir (line 2) or absolute
                - spec_file: <your_raw_ceria_data_directory>/spec.log
                  scan_numbers: 1
  - edd.MCATthCalibrationProcessor:
      config:
        calibration_method: direct_fit_bragg
        baseline: true
        energy_mask_ranges: [[81, 135]] # Change as needed
        detectors:  # The same as in the energy calibration when omitted
          - id: 0
          - id: 11
          - id: 22
        tth_initial_guess: 6.0
      save_figures: true

  # Create a CHAP style map
  - edd.EddMapReader:
      filename: <your_raw_data_directory>/id1a3-wbmapscan-s23-map-1.par
      dataset_id: 1
      schema: common.models.map.MapConfig
  - common.MapProcessor:
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
        baseline: true
        detectors: # Edit: Do not leave this list empty!
          - id: 0
          - id: 11
          - id: 22
        energy_mask_ranges: [[77, 110]] # Change as desired
        materials:
          - material_name: Al
            sgnum: 225
            lattice_parameters: 4.046
        rel_height_cutoff: 0.05         # Change as desired
        skip_animation: false
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

- `profile`: Runs the pipeline in a [Python profiler](https://docs.python.org/3/library/profile.html).

The "pipeline" block creates the actual workflow pipeline, it this example it consists of four toplevel processes that get executed successively:

- `common.MapProcessor`: A processor that creates a CHESS style map.

- `pipeline.MultiplePipelineItem`: A processor that executes (in this case reads) three items and passes the inputs on to the next item in the pipeline.

- `tomo.TomoCHESSMapConverter`: A processor that converts the inputs to a CHESS style map.

- `tomo.TomoDataProcessor`: The actual tomographic reconstruction processor that creates a single NeXus object with the reconstructed data as well as all metadata pertaining to the reconstruction and passes it on to the next item in the pipeline.

- `common.NexusWriter`: A processor that writes the reconstructed data to a NeXus file.



