# HDRM subpackage (CHAP.hdrm)

The HDRM subpackage contains the modules that are unique to High Dynamic Range Mapping (HDRM) data processing workflows. This document describes how to run the tools to stack the raw data, integrate the data azimuthally, find Bragg peaks, and obtain the orientation matrix (calibrating the detector & beamline and getting the HKLs is not yet implemented in CHAP).

## Activating the HDRM conda environment

### From the CHESS Compute Farm

Log in to the CHESS Compute Farm and activate the `CHAP_hdrm` environment:
```bash
source /nfs/chess/sw/miniforge3_chap/bin/activate
conda activate CHAP_hdrm
```

### From a local CHAP clone

1. Create and activate a base conda environent, e.g. with [Miniforge](https://github.com/conda-forge/miniforge).
1. Install a local version of the CHAP package according to the [instructions](/docs/installation.md)
1. Create the HDRM conda environment:
   ```bash
   mamba env create -f <path_to_CHAP_clone_dir>/CHAP/hdrm/environment.yml
   ```
1. Activate the `CHAP_hdrm` environment:
   ```bash
   conda activate CHAP_hdrm
   ```

## Running a HDRM workflow

1. Navigate to your work directory.
1. Create the required CHAP pipeline file for the workflow (see below) and any additional workflow specific input files. 
1. Run the workflow:
   ```bash
   CHAP <pipelinefilename>
   ```

## Inspecting output

The output consists of a single NeXus (`.nxs`) file containing the results of the analysis as well as all metadata pertaining to it. Additionally, optional output figures (`.png`) may be saved to an output directory specified in the pipeline file.

Any of the optional output figures can be viewed directly by any PNG image viewer. The data in the NeXus output file can be viewed in [NeXpy](https://nexpy.github.io/nexpy/), a high-level python interface to HDF5 files, particularly those stored as [NeXus data](http://www.nexusformat.org):

1. Open the NeXpy GUI by entering in your terminal:
   ```bash
   nexpy &
   ```
1. After the GUI pops up, click File-> Open to navigate to the folder where your output `.nxs` file was saved, and select it.
1. Navigate the filetree in the "NeXus Data" panel to inspect any output or metadata field.

## Creating the pipeline file

Create a workflow `pipeline.yaml` file according to the [instructions](/docs/pipeline.md). A generic pipeline input file is as follows (note that spaces and indentation are important in `.yaml` files):
```
config:
  root: .            # Change as desired
  inputdir: .        # Change as desired
  outputdir: .       # Change as desired
  interactive: false # None of these tools have interactive parts
  log_level: INFO
  profile: false

map:

  # Stack the raw detector image files
  - common.MapProcessor:
      config:
        station: id4b
        experiment_type: HDRM
        spec_scans: # Edit: spec.log path and scan numbers
                    # Path can be relative to inputdir (line 2) or absolute
          - spec_file: <your_raw_data_directory>/spec.log
            scan_numbers: 1 # Change as desired
        independent_dimensions:
        - label: phi
          units: degrees
          data_type: scan_column
          name: phi
        scalar_data:
        - label: chi
          units: degrees
          data_type: spec_motor
          name: chi
        - label: mu
          units: degrees
          data_type: spec_motor
          name: mu
        - label: eta
          units: degrees
          data_type: spec_motor
          name: th
      detectors:
        - id: PIL10
  - common.NexusWriter:
      filename: map.nxs # Change as desired
                        # will be placed in 'outdutdir' (line 3)
      force_overwrite: true # Do not set to false!
                            # Rename an existing file if you want to prevent
                            # it from being overwritten

integrate:

  # Integrate the raw detector image data
  - common.NexusReader:
      filename: map.nxs
  - giwaxs.PyfaiIntegrationProcessor:
      config:
        azimuthal_integrators: # Edit: PONI and mask file paths
          - id: PIL10
            poni_file: <path_to_poni_file_location>/basename.poni
                       # Path can be relative to inputdir (line 2) or absolute
            mask_file: <path_to_mask_file_location>/mask.edf
                       # Path can be relative to inputdir (line 2) or absolute
        integrations:
          - name: azimuthal
            integration_method: integrate1d
            integration_params:
              ais: PIL10
              azimuth_range: null
              radial_range: null
              unit: q_A^-1
              npt: 8000
        sum_axes: true # This will sum the data over the independent dimension
      save_figures: false
  - common.NexusWriter:
      filename: integrated.nxs # Change as desired
                               # will be placed in 'outdutdir' (line 3)
      force_overwrite: true # Do not set to false!
                            # Rename an existing file if you want to prevent
                            # it from being overwritten

peaks:

  # Find the Bragg peaks
  - common.NexusReader:
      filename: map.nxs
  - hdrm.HdrmPeakfinderProcessor:
      config:
        peak_cutoff: 0.95 # Change as desired
  - common.NexusWriter:
      filename: peaks.nxs   # Change as desired
                            # will be placed in 'outdutdir' (line 3)
      force_overwrite: true # Do not set to false!
                            # Rename an existing file if you want to prevent
                            # it from being overwritten

orm:

  # Solve for the orientation matrix
  - common.NexusReader:
      filename: peaks.nxs
  - hdrm.HdrmOrmfinderProcessor:
      config:
        azimuthal_integrators: # Edit: PONI and mask file paths
          - id: PIL10
            poni_file: <path_to_poni_file_location>/basename.poni
                        # Path can be relative to inputdir (line 2) or absolute
        materials:
          - material_name: FeNiCo      # Change as desired
            sgnum: 225                 # Change as desired
            lattice_parameters: 3.569  # Change as desired
  - common.NexusWriter:
      filename: orm.nxs     # Change as desired
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

The remaining blocks create the actual workflow pipeline, in this example it consists of four toplevel sub-workflows bracketed by their (optional) CHAP readers and writers. These sub-workflows can be get executed individually or in a certain combination, or they can all four be execute successively as a single workflow.

- Stacking the raw detector image files consists of one processor and a writer:

    - `common.MapProcessor`: A CHAP processor that reads the raw detector image data and collects everything in a single CHAP style map.

    - `common.NexusWriter`: A CHAP writer that writes the stacked data map to a NeXus file.

- Integrating the raw detector image data consists of one processor, optionally bracketed by a reader and writer:

    - `common.NexusReader`: A CHAP reader that reads the stacked data map from a NeXus file.

    - `giwaxs.PyfaiIntegrationProcessor`: A CHAP processor that performs azimuthal integration of the image data.

    - `common.NexusWriter`: A CHAP writer that writes the integrated image data to a NeXus file.

- Finding the Bragg peaks consists of one processor, optionally bracketed by a reader and writer:

    - `common.NexusReader`: A CHAP reader that reads the stacked data map from a NeXus file.

    - `hdrm.HdrmPeakfinderProcessor`: A CHAP processor that finds the Bragg peaks in the stacked image data.

    - `common.NexusWriter`: A CHAP writer that writes the peak information to a NeXus file.

- Solving for the orientation matrix consists of one processor, optionally bracketed by a reader and writer:

    - `common.NexusReader`: A CHAP reader that reads the peak information from a NeXus file.

    - `hdrm.HdrmOrmfinderProcessor`: A CHAP processor that obtains the orientation matrix from the Bragg peaks in the stacked image data.

    - `common.NexusWriter`: A CHAP writer that writes the orientation matrix to a NeXus file.

## Executing the pipeline file

The workflow pipeline can be executed as a single workflow, but, as mentioned above, the four toplevel sub-workflows can also be executed individually or in a certain combination. When the entire pipeline or several individual sub-workflows are executed, the enclosed pairs of CHAP readers and writers can be commented out or removed from the pipeline file. In this case each processor's output will be piped to the next processor as available input. This can greatly reduce the processing time and/or required memory to store the results.

Running the entire workflow pipeline is as described above under `Running a HDRM workflow`. To create only the stacked data map, run:
   ```bash
   CHAP <pipelinefilename> -p map
   ```
Do not remove or comment out the Nexus writer that writes the stacked data to file in this case!

If instead you would like to create the orientation matrix from the raw image data files, without performing the azimuthal integration, you can run:
   ```bash
   CHAP <pipelinefilename> -p map peaks orm
   ```
In this case the individual sub-workflows are added to the `-p` flag separated by spaces and in the correct order of processing. You can now comment out or remove the Nexus writer and reader for the stacked data map, as well as those for the Bragg peak information, but do not remove the final orientation matrix writer!

To create the azimuthally integrated data, run either:
   ```bash
   CHAP <pipelinefilename> -p map integrate
   ```
in which case you can optionally skip the intermediate writing of the stacked image stack, or run:
   ```bash
   CHAP <pipelinefilename> -p integrate
   ```
in which case you have to load the stacked image data map in the `integrate` sub-workflow from an earlier created map Nexus file.

Note that the each processor adds data and metadata to the loaded Nexus file. However, to reduce the total file size and since the orientation matrix processor only needs the Bragg peaks, the processor that finds the Bragg peaks will strip the raw data from the existing Nexus file.
Finally, note that the actual sub-workflow labels `map`, `integrate`, `peaks`, and `orm` are irrelevant to CHAP, the user is free to chose any single string of characters followed by a semi-colon as label for a sub-workflow. Just one label is required after the `config` block, i.e., at the start of the actual workflow pipeline.
