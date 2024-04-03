# Tomography subpackage (CHAP.tomo)

The tomography subpackage contains the modules that are unique to tomography
data processing workflows. This document describes how to run a tomography
reconstruction workflow in a Linux terminal.

A standard tomographic reconstruction in CHAP consists of three steps:

1. Reducing the data, i.e., correcting the raw detector images for background
and non-uniformities in the beam intensity profile using dark and bright fields
collected separately from the tomography image series.

1. Finding the calibrated rotation axis. Accurate reconstruction relies on accurately knowing the center of rotation at each data plane perpendicular to the rotation axis (the sinogram). This rotation axis is calibrated by selecting two data planes, one near the top and one near the bottom of the sample or beam, and visually or automatically picking the optimal center location.

1. Reconstructing the reduced data for the calibrated rotation axis. For samples taller than the height of the beam, this last step can consist of two parts:

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
1. Create the required CHAP pipeline file for the workflow (see below) and any additional workflow specific input files. This typically includes at a minimum the detector configuration `.yaml` file with the detector image size information. For example, for the andor2 detector (`andor2.yaml`):
    ```
    prefix: andor2
    rows: 2160
    columns: 2560
    pixel_size:
    - 0.0065
    - 0.0065
    lens_magnification: 5.0
    ``` 
    Here, the base file name and the prefix field must equal the detector name and mach any `detector_names` fields in the pipeline input file.
1. Run the reconstruction:
   ```bash
   CHAP <pipelinefilename>
   ```
1. Respond to any prompts that pop up if running interactively.

## Inspecting output

The output consists of a single NeXus (`.nxs`) file containing the reconstructed data set as well as all metadata pertaining to the reconstruction. Additionally, optional output figures (`.png`) may be save to an output directory specified in the pipeline file.

Any of the optional output figures can be viewed directly by any PNG image viewer. The data in the NeXus output file can be viewed in [NeXpy](https://nexpy.github.io/nexpy/), a high-level python interface to HDF5 files, particularly those stored as [NeXus data](http://www.nexusformat.org):

1. Open the NeXpy GUI by entering in your terminal:
   ```bash
   nexpy &
   ```
1. After the GUI pops up, click File-> Open to navigate to the folder where your output `.nxs` file was saved, and select it.
1. Double click on the base level `NXroot` field in the leftmost "NeXus Data" panel to view the reconstruction. Note that the `NXroot` name is always the basename of the output file.
1. Or navigate the filetree in the "NeXus Data" panel to inspect any other output or metadata field. Note that the latest data set in any tomography reconstruction workflow is always available under the "data" `NXdata` field among the default `NXentry`'s fields (it is this data set that is opened in the viewer panel when double clicking the `NXroot` field). The default `NXentry` name is always the "title" field in the workflow's map configuration.

## Creating the pipeline file

Create a workflow `pipeline.yaml` file according to the [instructions](/docs/pipeline.md). A generic pipeline input file for a full tomography reconstruction workflow is as follows (note that spaces and indentation are important in `.yaml` files):
```
config:
  outputdir: output # Change as desired
  interactive: true # Change as desired
  log_level: INFO
  profile: false

pipeline:

  # Create the map
  - pipeline.MultiplePipelineItem:
      items:
        - common.SpecReader:
            spec_config:
              station: id3a # Change as needed
              experiment_type: TOMO
              spec_scans: # Edit: absolute spec.log path and dark field scan number
              - spec_file: <your_raw_data_directory>/spec.log
                scan_numbers:
                - 1
            detector_names:
              - andor2 # Change as needed
            schema: darkfield
        - common.SpecReader:
            spec_config:
              station: id3a # Change as needed
              experiment_type: TOMO
              spec_scans: # Edit: absolute spec.log path and bright field scan number
              - spec_file: <your_raw_data_directory>/spec.log
                scan_numbers:
                - 2
            detector_names:
              - andor2 # Change as needed
            schema: brightfield
        - common.MapReader:
            map_config:
              title: <your_BTR> # Change as desired, typically BTR
              station: id3a # Change as needed
              experiment_type: TOMO
              sample:
                name: <your_sample_name> # Change as desired
                                         # typically the sample name
              spec_scans: # Edit: absolute spec.log path and tomography scan number(s)
              - spec_file: <your_raw_data_directory>/spec.log
                scan_numbers: [3, 4, 5]
              independent_dimensions:
              - label: rotation_angles
                units: degrees
                data_type: scan_column
                name: theta
              - label: x_translation
                units: mm
                data_type: smb_par
                name: ramsx # Change as needed
              - label: z_translation
                units: mm
                data_type: smb_par
                name: ramsz # Change as needed
            detector_names:
              - andor2 # Change as needed
            schema: tomofields
        - common.YAMLReader:
            filename: andor2.yaml # Detector config file
                                  # Must be a relative or absolute path or
                                  # present in work directory
            schema: tomo.models.Detector
  - tomo.TomoCHESSMapConverter

  # Run the tomography reconstruction
  - tomo.TomoDataProcessor:
      reduce_data: True
      find_center: True
      reconstruct_data: True
      combine_data: True
      outputdir: saved_figs # Change as desired, unless an absolute path
                            # this will appear under 'outdutdir' on line 2
      save_figs: 'only'
  - common.NexusWriter:
      filename: reconstructed_sample.nxs # Change as desired
                                         # will be placed in 'outdutdir' on line 2
      force_overwrite: true # Do not set to false!
                            # Rename an existing file if you want to prevent
                            # it from being overwritten
```

## Example

The CHAP tomography subpackage comes with several workflow examples, one of them for an CHESS ID3A beamline style experiment of a truncated hollow four sided pyramid made from a single homogeneous material.

This example uses simulated raw imaging data that needs to be available in a specific location ahead of the reconstruction. If you are logged in on the CHESS Compute Farm, replace `<path_to_CHAP_clone_dir>` below with `/nfs/chess/sw/ChessAnalysisPipeline`, the path to the CHAP repository administrated by the CHAP developers. If not, replace it with the path to your local CHAP repository. In the later case, you will also need to create the raw data once, since it is not part of the cloned repository. To do so, create and activate your local CHAP_tomo conda environment as instructed above, navigate to your local CHAP repository, and execute:
```bash
CHAP examples/tomo/pipeline_id3a_pyramid_sim.yaml
```

To perform the reconstruction:

1. Create a work directory in your own user space.
1. Within the work directory, create a plain text file, named `pipeline_id3a_pyramid.yaml`, with the following content (note that spaces and indentation are important in `.yaml` files):
    ```
    config:
      root: .
      inputdir: <path_to_CHAP_clone_dir>/examples/tomo/config
      outputdir: hollow_pyramid
      interactive: true
      log_level: INFO
      profile: false

    pipeline:

      # Convert the CHESS style map
      - pipeline.MultiplePipelineItem:
          items:
            - common.SpecReader:
                spec_config:
                  station: id3a
                  experiment_type: TOMO
                  spec_scans:
                  - spec_file: ../data/hollow_pyramid/spec.log
                    scan_numbers:
                    - 1
                detector_names:
                  - sim
                schema: darkfield
            - common.SpecReader:
                inputdir: ../data/hollow_pyramid
                spec_config:
                  station: id3a
                  experiment_type: TOMO
                  spec_scans:
                  - spec_file: spec.log
                    scan_numbers:
                    - 2
                detector_names:
                  - sim
                schema: brightfield
            - common.MapReader:
                filename: map_id3a_pyramid.yaml
                detector_names:
                  - sim
                schema: tomofields
            - common.YAMLReader:
                filename: detector_pyramid.yaml
                schema: tomo.models.Detector
      - tomo.TomoCHESSMapConverter

      # Full tomography reconstruction
      - tomo.TomoDataProcessor:
          reduce_data: True
          find_center: True
          reconstruct_data: True
          combine_data: True
          outputdir: saved_figs
          save_figs: 'only'
      - common.NexusWriter:
          filename: combined_hollow_pyramid.nxs
          force_overwrite: true
    ```
1. Execute
    ```bash
    CHAP pipeline_id3a_pyramid.yaml
    ```
1. Follow the interactive prompts or replace `true` with `false` on line 5 in `pipeline_id3a_pyramid.yaml` (`interactive: false`) and run the workflow non-interactively.
1. Inspect the results:
    - In NeXpy as instructed above, navigate to `<your_work_directory>/hollow_pyramid` and open `combined_hollow_pyramid.nxs`
    - By displaying the output figures in `<your_work_directory>/hollow_pyramid/save_figs`



