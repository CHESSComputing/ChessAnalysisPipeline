(giwaxs_workflow)=
# GIWAXS subpackage (`CHAP.giwaxs`)

The GIWAXS subpackage contains the modules that are unique to Grazing Incidence Wide Angle X-ray Scattering (GIWAXS) data processing workflows. This document describes how to run a GIWAXS analysis workflow in a Linux terminal.

## Processing the data

A standard GIWAXS analysis in CHAP consists of three steps:

- Performing a flux correction to correct for variations in the incoming beam intensity and flat field.

- Performing the "missing wedge" correction to address missing regions in the reciprocal space due to the geometry of the sample holder and detector by mapping the detector data onto the space spanned by the in-plane and out-of-plane scattering vectors, $q_{xy}$ or $q_\parallel$ and $q_{z}$ or $q_\perp$, respectively.

- Performing additional integrations, like radial or cake integration.

## Creating and activating the GIWAXS conda environment (requires a local CHAP clone)

1. Create and activate a base conda environent, e.g. with [Miniforge](https://github.com/conda-forge/miniforge).
1. Install a local version of the CHAP package according to the [installation instructions](installation).
1. Create the GIWAXS conda environment:
   ```bash
   mamba env create -f <path_to_CHAP_clone_dir>/CHAP/giwaxs/environment.yml
   ```
1. Activate the `CHAP_giwaxs` environment:
   ```bash
   conda activate CHAP_giwaxs
   ```

## Running an GIWAXS workflow

1. Navigate to your work directory.
1. Create the required CHAP pipeline file for the workflow (see below) and any additional workflow specific input files. 
1. Run the workflow using your own `CHAP_giwaxs` conda environment:
   ```bash
   CHAP <pipelinefilename>
   ```
   or run the workflow using the latest production release version:
   ```bash
   /nfs/chess/sw/CHESS-software-releases/prod/CHAP_giwaxs <pipelinefilename>
   ```
   or the latest development release version:
   ```bash
   /nfs/chess/sw/CHESS-software-releases/dev/CHAP_giwaxs <pipelinefilename>
   ```
   You may find it convenient to add an alias to your `~/.bascrc` or `~/.bash_aliases`, for example for the CHAP GIWAXS production release:
   ```bash
   alias CHAP_giwaxs_prod='/nfs/chess/sw/CHESS-software-releases/prod/CHAP_giwaxs'
   ```
1. Respond to any prompts that pop up if running interactively.

## Inspecting output

The output consists of a single NeXus (`.nxs`) file containing the strain analysis data as well as all metadata pertaining to the analysis. Additionally, optional output figures (`.png`) may be saved to an output directory specified in the pipeline file.

The optional output figures can be viewed directly by any PNG image viewer. The data in the NeXus output file can be viewed in [NeXpy](https://nexpy.github.io/nexpy/), a high-level python interface to HDF5 files, particularly those stored as [NeXus data](http://www.nexusformat.org):

1. Open the NeXpy GUI by entering in your terminal:
   ```bash
   /nfs/chess/sw/nexpy/anaconda/envs/nexpy/bin/nexpy &
   ```
   You may find it convenient to add an alias to your `~/.bascrc` or `~/.bash_aliases`:
   ```bash
   alias nexus='/nfs/chess/sw/nexpy/anaconda/envs/nexpy/bin/nexpy &'
   ```
1. After the GUI pops up, click File-> Open to navigate to the folder where your output `.nxs` file was saved, and select it.
1. Navigate the filetree in the "NeXus Data" panel to inspect any output or metadata field.

(giwaxs_pipeline)=
## Creating the pipeline file

Create a workflow `pipeline.yaml` file according to the [CHAP pipeline instructions](chap_pipeline). A generic pipeline input file for a GIWAXS data reduction and cake integration is as follows:
```yaml
config:
  root: .                   # Change as desired
  inputdir: .               # Change as desired
                            # Path can be relative to root (line 2) or absolute
  outputdir: output         # Change as desired
                            # Path can be relative to root (line 2) or absolute
  interactive: true         # Change as desired
  log_level: info           # Set to debug, info, warning, or error

map:

  - common.MapProcessor:
      config:
        title: <your_BTR>   # Change as desired, typically BTR
        station: id3b # Change as needed
        experiment_type: GIWAXS
        sample:
          name: <your_sample_name> # Change as desired
                                   # typically the sample name
        spec_scans: # Edit both SPEC log file path and tomography scan numbers
                    # Path can be relative to inputdir (line 3) or absolute
          - spec_file: <your_raw_data_directory>/spec.log
            scan_numbers: [1,2,3,4]
        independent_dimensions:
        - label: theta
          units: degrees
          data_type: scan_column
          name: GI_samth    # Change as needed
        dwell_time_actual:
          data_type: scan_column
          name: mcs0        # Change as needed
        presample_intensity:
          data_type: scan_column
          name: ic3         # Change as needed
      detector_config:
        detectors:
          - id: PIL5        # Detector prefix, change as needed
  - common.NexusWriter:
      filename: map.nxs     # Change as desired
                            # will be placed in 'outdutdir' (line 5)
     force_overwrite: true  # Do not set to false!
                            # Rename an existing file if you want to prevent
                            # it from being overwritten

flux:

  - common.NexusReader:
      filename: map.nxs     # Map output filename, same as written to above
      nxpath: <title>/data/PIL5
      name: intensity
  - common.NexusReader:
      filename: map.nxs     # Map output filename, same as written to above
      nxpath: <title>/scalar_data/presample_intensity
      name: presample_intensity
  - common.NexusReader:
      filename: map.nxs     # Map output filename, same as written to above
      nxpath: <title>/scalar_data/dwell_time_actual
      name: dwell_time_actual
  - saxswaxs.FluxCorrectionProcessor:
      nxprocess: true
  - common.NexusWriter:
      filename: map.nxs
      nxpath: /<title>_flux_corrected
      force_overwrite: true

convert:

  - common.NexusReader:
      filename: map.nxs     # Map output filename, same as written to above
  - giwaxs.GiwaxsConversionProcessor:
      nxpath: <title>_flux_corrected/data/result
      config:
        azimuthal_integrators:
          - id: PIL5        # Detector prefix, change as needed
            poni_file: <your_raw_data_directory>/<poni_filename>
                    # Edit the poni file filename and path
                    # Path can be relative to inputdir (line 3) or absolute
        integrations:
          - name: wedge
            integration_method: integrate2d_grazing_incidence
            integration_params:
              ais: PIL5     # Detector prefix, change as needed
              npt_ip: 400   # Number of in-plane points after integration
              npt_oop: 400  # Number of out-of-plane points after integration
  - common.NexusWriter:
      filename: giwaxs.nxs  # Change as desired, unless an absolute path
                            # this will appear under 'outdutdir' (line 5)
      force_overwrite: true # Do not set to false!
                            # Rename an existing file if you want to prevent
                            # it from being overwritten

integrate:

  - common.NexusReader:
      filename: giwaxs.nxs  # Wedge corrected output filename as above
  - giwaxs.PyfaiIntegrationProcessor:
      config:
        azimuthal_integrators:
          - id: PIL5        # Detector prefix, change as needed
            poni_file: <your_raw_data_directory>/<poni_filename>
                    # Edit the poni file filename and path
                    # Path can be relative to inputdir (line 3) or absolute
        integrations:
          - name: cake
            integration_method: integrate2d
            multi_geometry:
              ais: PIL5     # The detector to be integrated
              azimuth_range: [-180.0, 180.0] # Change as desired
              radial_range: [0, 1.5] # Change as desired
              unit: q_A^-1
            integration_params:
              npt_azim: 360 # Change as desired
              npt_rad: 180  # Change as desired
  - common.NexusWriter:
      filename: giwaxs.nxs
      force_overwrite: true
```

The "config" block defines the CHAP generic configuration parameters:

- `root`: The work directory, defaults to the current directory (where `CHAP <pipelinefilename>` is executed). Must be an absolute path or relative to the current directory.

- `inputdir`: The default directory for files read by any CHAP reader (must have read access), defaults to `root`. Must be an absolute path or relative to `root`.

- `outputdir`: The default directory for files written by any CHAP writter (must have write access, will be created if not existing), defaults to `root`. Must be an absolute path or relative to `root`.

- `interactive`: Allows for user interactions, defaults to `False`.

- `log_level`: The [Python logging level](https://docs.python.org/3/library/logging.html#levels).

The remainder of the file contains the actual workflow pipeline, in this example it consists of four blocks, `map`, `flux`, `convert`, and `integrate`, which can be executed individually or all at once [as described here](chap_pipeline). In addition to the NeXus reader and writer of the intermediate results (`common.NexusReader` and `common.NexusWriter`, respectively), four toplevel processors get executed successively in the combined four pipeline blocks:

- Creating the Map representing the experimental data: the raw data, independent dimensions and detector information:

    - `common.MapProcessor`: A processor create a CHAP style raw data map.

- Flux correction:

    - `saxswaxs.FluxCorrectionProcessor`: A processor that performs the flux correction.

- Wedge correction:

    - `giwaxs.GiwaxsConversionProcessor`: A processor performing the wedge correction and conversion to in-plane and out-of-plane scattering vectors.

- Cake integration:

    - `giwaxs.PyfaiIntegrationProcessor:`: The processor performing the cake integration (or any of he other optional integrations).
