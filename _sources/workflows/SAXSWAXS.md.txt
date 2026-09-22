(saxswaxs_workflow)=
# SAXS/WAXS module (`CHAP.saxswaxs`)

The `CHAP.saxswaxs` module contains processing tools unique to SAXS/WAXS processing. This document describes how to use them in a pipeline configuration YAML file so that SAXS/WAXS workflows can be run from the terminal on the CHESS Linux system.


## Workflow description

### General overview

1. Setup a [Zarr](https://zarr.dev) container for the complete raw and processed dataset.

   First, we set up a container to hold the raw and processed datasets before any processing begins. This step needs the following information to allocate an appropriate amount of space:

   1. Dataset size
   1. Detector size(s)
   1. Integrated data size(s)

   In practice, you should prepare [two supplementary configuration files](#related_configuration_objects) in addition to the parameters for the tools involved in this step: one for a `MapConfig`, object, another for a `PyFaiIntegrationProcessorConfig` object. 
   This step also sets the number of chunks for each data array in the container. [Selecting the right number of chunks is important for optimizing performance](#optimizing-performance) during the next step.

    <details>
    <summary> Example pipeline configuration</summary>

   ```yaml
   config:
     root: .
     log_level: debug
   setup:
   # Step 1: read in configuration files
   - common.YAMLReader:
       filename: map_config.yaml
       schema: common.models.map.MapConfig
   - common.YAMLReader:
       filename: pyfai_integration_processor_config.yaml
       schema: common.models.integration.PyfaiIntegrationConfig
   # Step 2: set up Zarr container for processed data based on
   #         configurations provided from Step 1
   - saxswaxs.SetupProcessor:
       dataset_shape:
       - 50
       dataset_chunks:
       - 10
       detectors:
       - id: PIL9
	 shape:
	 - 407
	 - 487
       - id: PIL11
	 shape:
	 - 407
	 - 487
   # Step 3: Write the Zarr container to file
   - common.ZarrWriter:
       filename: data.zarr
       force_overwrite: true   
   ```
    </details>
   
1. Fill container with processed data

   Next, we read in the raw data and perform the configured integration(s). To get the best performance for large datasets, this step should be performed across multiple pipeline processes runnning in parallel. Each process will handle reading, processing, and writing just one part of the whole dataset -- exactly _one chunk_ of each array in the container set up previously, to be precise. To fill the container from the previous step with processed data, each parallel process will need to know the following information:

   1. The spec file for the scan whose data will be processed
   1. The scan number for the scan whose data will be processed
   1. Parameters indicating a specific slice of the scan to process

    <details>
    <summary> Example pipeline configuration</summary>

   ```yaml
   config:
     root: .
     log_level: debug
   update:
   # Step 1: read in configuration files
   - common.YAMLReader:
       filename: map_config.yaml
       schema: common.models.map.MapConfig
   - common.YAMLReader:
       filename: pyfai_integration_processor_config.yaml
       schema: common.models.integration.PyfaiIntegrationConfig
   # Step 2: read and process the data
   - saxswaxs.UpdateValuesProcessor:
       spec_file: spec_file
       scan_number: 1
       detectors:
       - id: PIL9
         shape:
         - 407
         - 487
       - id: PIL11
         shape:
         - 407
         - 487
       idx_slice:
         start: 0
         stop: 10
         step: 1
   # Step 3: write processed data to the container set up earlier
   - common.ZarrValuesWriter:
       filename: data.zarr
   ```
     </details>

1. Perform final user adjustments

   These "final adjustments" are all optional, but they can include the following:
   - converting from .zarr format to NeXus format
   - performing flux, absorption, and / or background corrections on integrated data
   - inserting links to coordinate axes next to processed data arrays
   - reshaping the dataset from an "unstructured" to a "structured" representation

    <details>
    <summary> Example format conversion pipeline configuration</summary>

   ```yaml
   config:
     root: .
     log_level: debug
   convert:
   # One tool only: ZarrToNexusProcessor takes care of reading old
   # zarr file and writing new NeXus file, too
   - common.ZarrToNexusProcessor:
       zarr_filename: data.zarr
       nexus_filename: data.nxs
   ```
    </details>

    <details>
    <summary> Example flux correction pipeline configuration</summary>

   ```yaml
   config:
     root: .
     log_level: debug
   flux_correct:
   # Step 1: Read in uncorrected data and flux data to perform flux correction
   - common.NexusReader:
       filename: data.nxs
       nxpath: waxs_azimuthal/data/I
       nxmemory: 100000
       name: intensity
   - common.NexusReader:
       filename: data.nxs
       nxpath: spec_file_012/scalar_data/presample_intensity
       nxmemory: 100000
       name: presample_intensity
   # Step 2: Perform flux correction calculations
   - saxswaxs.FluxCorrectionProcessor:
       nxprocess: true
       presample_intensity_reference_rate: 50000
   # Write flux corrected results back to original file as their own
   # new NXprocess group
   - common.NexusWriter:
       filename: data.nxs
       nxpath: /waxs_azimuthal_flux_corrected
       force_overwrite: true
   ```
    </details>

    <details>
    <summary> Example coordinate linking pipeline configuration</summary>

   ```yaml
   config:
     root: .
     log_level: debug
   linkdims:
   # Step 1: Read in the Nexus file in which we're adding linked data fields
   - common.NexusReader:
       filename: data.nxs
       mode: rw
   # Step 2: Create new linked data arrays from all the groups listed
   # in `link_from` to all the existing data arrays listed in
   # `link_to`
   - common.NexusMakeLinkProcessor:
       link_from:
       - waxs_azimuthal/data
       - waxs_cake/data
       - waxs_radial/data
       - waxs_azimuthal_flux_abs_bg_corrected/data
       - waxs_azimuthal_flux_abs_corrected/data
       - waxs_azimuthal_flux_corrected/data
       - waxs_cake_flux_abs_bg_corrected/data
       - waxs_cake_flux_abs_corrected/data
       - waxs_cake_flux_corrected/data
       - waxs_radial_flux_abs_bg_corrected/data
       - waxs_radial_flux_abs_corrected/data
       - waxs_radial_flux_corrected/data
       link_to:
       - spec_file_012/independent_dimensions/samx
       - spec_file_012/independent_dimensions/samzmakelink
   # No writing tool necessary, `common.NexusMakeLinkProcessor` will
   # modilfy the file in place as long as it's read in with `mode:
   # rw`.
   ```
    </details>

    <details>
    <summary> Example unstructured-to-structured pipeline configuration</summary>

   ```yaml
   config:
     root: .
     log_level: debug
   struct:
   # Step 1: Read in the data arrays to structure, and all their
   # coordinate axes arrays too
   - common.NexusReader:
       filename: data.nxs
       nxpath: spec_file_012/independent_dimensions/samx
       name: samx
   - common.NexusReader:
       filename: data.nxs
       nxpath: spec_file_012/independent_dimensions/samz
       name: samz
   - common.NexusReader:
       filename: data.nxs
       nxpath: waxs_azimuthal/data/q_A^-1
       name: waxs_azimuthal_q
   - common.NexusReader:
       filename: data.nxs
       nxpath: waxs_radial/data/chi_deg
       name: waxs_radial_chi
   - common.NexusReader:
       filename: data.nxs
       nxpath: waxs_azimuthal/data/I
       name: waxs_azimuthal_I
   - common.NexusReader:
       filename: data.nxs
       nxpath: waxs_radial/data/I
       name: waxs_radial_I
   # Step 2: restructure signal arrays in a single shared NXdata object
   - saxswaxs.UnstructuredToStructuredProcessor:
       name: structured_data
       fields:
       - name: samx
         type: axis
       - name: samz
         type: axis
       - name: waxs_azimuthal_q
         type: axis
       - name: waxs_radial_chi
         type: axis
       - name: waxs_azimuthal_I
         axes: [samx, samz, waxs_azimuthal_q]
         type: signal
       - name: waxs_radial_I
         axes: [samx, samz, waxs_radial_chi]
         type: signal
   # Step 3: Write the structured data to a new group in the original NeXus file
   - common.NexusWriter:
       filename: data.nxs
       nxpath: /structured
       force_overwrite: true
   ```
    </details>
### Output data formats
- This is a basic guide to help find the most essential datasets produced in SAXS/WAXS analysis. It is not an exhaustive specification of every last bit of data / metadata that one can possibly find in the data formats in question.
- The `CHAP.saxswaxs.processor.SetupProcessor` determines a hierarchical tree structure to contain processed SAXSWAXS data.
- This tree structure is returned in [Zarr](https://zarr.dev) file format to start, but it may be converted:
    - to the same format of NeXus file as that produced by [saxswaxsworkflow](https://gitlab01.classe.cornell.edu/msn-c/saxswaxsworkflow) using `CHAP.saxswaxs.processor.NewZarrToOldNexusProcessor`
    - to an indentically structured [NeXus](https://manual.nexusformat.org/index.html) file in a reversible operation using `CHAP.common.processor.ZarrToNexusProcessor` (reversible with `CHAP.common.processor.NexusToZarrProcessor`).

#### `CHAP.saxswaxs.processor.SetupProcessor` output structure
##### Root level
- The structure of the Zarr container is related to the configuration fields provided to `CHAP.saxswaxs.processor.SetupProcessor`:
    - `map_config` defines the parameters needed to locate and read in a raw CHESS dataset
    - `pyfai_config` defines one or more [pyFAI](https://pyfai.readthedocs.io/en/stable/) integrations to perform on 2D detector data (raw or corrected)
    - `correction_config` defines any number of standard SAXS/WAXS corrections to perform (on raw or integrated data)
    - `fit_config` defines any number of fits to perform on processed 1D data (integrated and corrected, or just integrated)
    - `map_config` and each individual integration / correction / fit defined in `pyfai_config`/`correction_config`/`fit_config` will correspond to a root-level group of the results container.
        - The name of the group is determined by the configuration's `title` field.
        - For instance, suppose `CHAP.saxswaxs.processor.SetupProcessor` is run with the following configurations:
          ```
          map_config:
            title: my_dataset
            <...>
          pyfai_config:
            - title: saxs_radial
              <...>
          correction_config:
            - title: saxs_radial_corrected
              <...>
          fit_config:
            - title: fit_saxs_radial
              <...>
            - title: fit_saxs_radial_corrected
              <...>
          ```
            ...then the resulting tree hierarchy will be:
          ```
          /                                 # root level group
              my_dataset/                   # raw data group
                  <...>
              saxs_radial/                  # integrated data group
                  <...>
              saxs_radial_corrected/        # corrected data group
                  <...>
              fit_saxs_radial/              # fit data group
                  <...>
              fit_saxs_radial_corrected/    # fit data group
                  <...>
          ```
    - NB: the types and names of the root level groups returned by `CHAP.saxswaxs.processor.SetupProcessor` are the same as those in the NeXus file produced by `saxswaxsworkflow`.
    - Descriptions of the trees found underneath each root-level group are below.

##### `map_config` sub-tree
- This structure is generic, not specific to `CHAP.saxswaxs`.
- The tree underneath looks like:
  ```
  <map_config.title>/
      map_config                    # map configuration serialized as JSON string
      detector_config               # detector configuration serialized as JSON string
      spec_scans/                   # metadata about each SPEC scan in the map
      data/                         # group containing data arrays of raw detector frames, one dataset per detector
      <sample_name>/                # sample metadata
      independent_dimensions/       # group containing data arrays of the map's coordinate values, one dataset per dimension
      scalar_data/                  # group containing data arrays of scalar-valued data, one dataset per scalar readback channel
                                    #   (includes beam intensity monitors, dwell time, etc.)
      detector_ids                  # list of detector identifiers
  ```
- Dimensions that are constant across the map are demoted from `independent_dimensions/` into `scalar_data/`.

##### `pyfai_config` sub-trees
- There is one root-level group per integration defined in `pyfai_config` (identified by each integration's `title` / `name` field).
- The tree underneath looks like:
  ```
  <integration_name>/
      data/
          I                             # integrated intensity array
          <radial_or_azimuthal_coord>   # coordinate axis array (e.g. q values, chi angles)
          [<azimuthal_or_radial_coord>] # other coordinate axis array (if the integration is 2D)
  ```
- The coordinate array name and units depend on the integration method:
    - `integrate1d`: one radial coordinate (e.g. `q_A^-1`)
    - `integrate2d`: one azimuthal coordinate + one radial coordinate
    - `integrate_radial`: one azimuthal coordinate (e.g. `chi_deg`)
- The `data/` group also carries links back to the map coordinate arrays in the `map_config` group.

##### `correction_config` sub-trees
- There is one root-level group per correction defined in `correction_config` (identified by each correction's `title` / `name` field).
- The tree underneath looks like:
  ```
  <correction_name>/
      data/
          I_corrected               # corrected intensity (single upstream source)
          # — or, if multiple upstream sources are named —
          I_corrected_<source>      # one corrected-intensity array per source
          # — when a background scan is configured —
          I_background              # averaged background intensity frame
          background_presample_intensity
          background_postsample_intensity
  ```
- The `data/` group carries links back to the map coordinate arrays in the `map_config` group and to the coordinate arrays of the upstream integration.
##### `fit_config` sub-trees
- There is one root-level group per fit defined in `fit_config` (identified by each fit's `title` / `name` field).
- The tree underneath looks like:
  ```
  <fit_name>/
      data/
          best_fit                  # best-fit curve at each scan point
          residual                  # residual (data minus best fit) at each scan point
          redchi                    # reduced chi-squared at each scan point
          num_func_eval             # number of function evaluations at each scan point
          success                   # whether the fit converged at each scan point
      components/
          <model_name>/             # one sub-group per model component (e.g. "gaussian", "linear")
              parameters/           # one sub-group per fit parameter
                  <param_name>/
                      value         # fitted value at each scan point
                      error         # uncertainty at each scan point
                      initial       # initial value
                      min, max      # bounds
                      vary          # whether the parameter was free
                      expression    # constraint expression, if any
              data/
                  best_fit          # component's contribution to the best-fit curve
  ```
- The `data/` group carries links back to the map coordinate arrays in the `map_config` group and to the coordinate arrays of the upstream integration or correction.

#### `CHAP.saxswaxs.processor.NewZarrToOldNexusProcessor` output structure
- Converts a Zarr file created by `CHAP.saxswaxs.processor.SetupProcessor` into an `NXroot` object whose layout mimics the NeXus files produced by [saxswaxsworkflow](https://gitlab01.classe.cornell.edu/msn-c/saxswaxsworkflow) and is compatible with the fitting notebook tools in the same repository.
- The output NeXus file contains two kinds of top-level groups:
    - One `NXentry` containing the map's coordinate values and scalar-valued data
    - One `NXprocess` per integration or correction, containing the processed data
- All data arrays in this structure are reshaped to be the same shape of the map (as opposed to flattened along the map's dimenisions, as they are in the Zarr format above).

##### `map_config` sub-tree
- This structure is nearly identical to the other `map_config` sub-tree defined above, but with some important differences:
    - the group containing arrays of raw detector data is not present
    - coordinate value arrays contain only the _unique, sorted_ values along each dimension (which may or may not be smaller than the length of the full dataset)

##### `pyfai_config` sub-trees
- There is one `NXprocess` group per integration in the Zarr file.
- The tree underneath looks like:
  ```
  <integration_name>/
      data/
          I                             # integrated intensity array
          <radial_or_azimuthal_coord>   # intensity coordinate axis array (e.g. q values, chi angles)
          [<azimuthal_or_radial_coord>] # other intensity coordinate axis array (if the integration is 2D)
          <dim_label>                   # map coordinate axis array (links to /<map_title>/independent_dimensions/<dim_label>)
          [<other_dim_label>]           # more links to /<map_title>/independent_dimensions/<other_dim_label> if the map has more than one independent_dimension
  ```
- Each `data/` group carries `axes` and `*_indices` attributes so that every axis — both map dimensions and signal coordinate axes — is labelled.

##### `correction_config` sub-trees
- There is one `NXprocess` group per correction in the Zarr file.
- The tree underneath looks like:
  ```
  <correction_name>/
      data/
          I_corrected                   # corrected intensity array
          I_uncorrected                 # uncorrected intensity source array (links to /<uncorrected_data_name>/data/I)
          <radial_or_azimuthal_coord>   # intensity coordinate axis array (e.g. q values, chi angles)
          [<azimuthal_or_radial_coord>] # other intensity coordinate axis array (if the integration is 2D)
          <dim_label>                   # map coordinate axis array (links to /<map_title>/independent_dimensions/<dim_label>)
          [<other_dim_label>]           # more links to /<map_title>/independent_dimensions/<other_dim_label> if the map has more than one independent_dimension
   ```
- Each `data/` group carries `axes` and `*_indices` attributes so that every axis — both map dimensions and signal coordinate axes — is labelled.


### Related Configuration Objects
Before constructing a `CHAP` pipeline configuration to run a complete SAXS/WAXS data processing workflow, users should first assemble two supplementary configuration files. *Both* these configurations will need to be read in to the pipeline for [the setup step](#setup) and every [update step](#update).
1. `map_config.yaml` (configuring a `CHAP.common.models.map.MapConfig`)
   
   This configuration contains everything `CHAP` needs to know about the location, format, and size of the raw input dataset.
   <details>
   <summary>Example `map_config.yaml`</summary>

   ```yaml
   validate_data_present: false
   title: spec_file_012
   station: id3b
   experiment_type: SAXSWAXS
   sample:
     name: spec_file
   spec_scans:
   - spec_file: spec_file
     scan_numbers:
     - 12
   independent_dimensions:
   - label: samx
     units: unknown units
     data_type: spec_motor
     name: samx
   - label: samz
     units: unknown units
     data_type: spec_motor
     name: samz
   dwell_time_actual:
     data_type: scan_column
     name: mcs0
   presample_intensity:
     data_type: scan_column
     name: ic3
   scalar_data: []
   postsample_intensity:
     data_type: scan_column
     name: diode
   ```
   </details>

   <details>
   <summary>Annotated `map_config.yaml` template</summary>

    ```yaml
    # General  metadata ####################################################

    # A title for this map. Recommended: use snake_case.
    title:

    # The station at which this map's data were taken. Choices: id1a3,
    # id3a, or id3b.
    station: 

    # Do not change this value!
    experiment_type: SAXSWAXS

    ########################################################################


    # Sample metadata ######################################################

    sample:
      # A name for the sample. 
      name: 

      # Optional: a free-text description of the sample.
      description: 

    ########################################################################

    # Location of SPEC scans that make up the map ##########################

    # true or false. Use false for processing live data.
    validate_data_present: false


    # There must be at least one item in this list. There is no maximum
    # number of items that can be in this list.
    spec_scans:

    # This is the first item in the list (required):
    - 
      # The full path to a spec file containing scans for this map.
      spec_file: /nfs/chess/raw/<cycle>/<station>/<BTR>/<spec_file>

      # A list of scan numbers from the above spec file that are part of
      # this map.
      scan_numbers: []

    # This is another additional item in the list (optional):
    - spec_file:
      scan_numbers: []

    ########################################################################


    # Location of data that represent each axis of the map #################

    # There must be at least one item in this list. There is no maximum
    # number of items that can be in this list.
    independent_dimensions:

    # This is the first item in the list (required):
    - 
      # A label for this axis. Recommended: use snake_case
      label: 

      # How values were recorded in the raw data files. Choices:
      # spec_motor, scan_column, smb_par, expression (for expressions
      # involving values from one or more of the data streams configured
      # in scalar_data), or scan_start_time. smb_par is only a valid
      # choice if data for this map was collected at station id1a3 or
      # station id3a.
      data_type:

      # The units in which values were recorded. If data_type is
      # scan_start_time, this field is optional and any value provided for
      # it will be ignored.
      units:

      # For data_type == spec_motor: the SPEC motor's mnemonic.
      # For data_type == scan_column: the SPEC data column label.
      # For data_type == smb_par: the name of the .par file column (found
      # in the corresponding .json file).
      # For data_type == expression: a mathematical expression that may
      # use the labels of any items in the scalar_data field (found below)
      # the built-in python function, `round()`, and any numpy function
      # (for example: `np.round` or `numpy.round`. Example usage: an
      # independent dimension of a map is the sum of two spec motors'
      # values. Both spec motors are configured as items in scalar_data
      # for the map, and have labels m1 and m2. Here, name could be:
      # round(m1 + m2, 4)
      # For data_type == scan_start_time: this field is optional and any
      # value provided for it will be ignored.
      name: 

    # This is an additional item in the list (optional):
    - label: 
      units: 
      data_type: 
      name: 

    ########################################################################


    # Location of data that will be used in corrections ####################

    # Required for all types of corrections:
    presample_intensity: 
      # How values were recorded in the raw data files. Choices:
      # scan_column, smb_par, or expression. smb_par is only a valid
      # choice if data for this map was collected at station id1a3 or
      # station id3a.
      data_type:

      # For data_type == spec_motor: the SPEC motor's mnemonic.
      # For data_type == scan_column: the SPEC data column label.
      # For data_type == smb_par: the name of the .par file column (found
      # in the corresponding .json file).
      # For data_type == expression: a mathematical expression that may
      # use the labels of any items in the scalar_data field (found below)
      # and the built-in python function, `round()`. Example usage: an
      # independent dimension of a map is the sum of two spec motors'
      # values. Both spec motors are configured as items in scalar_data
      # for the map, and have labels m1 and m2. Here, name could be:
      # round(m1 + m2, 4)
      name:

    # Required for all types of corrections:
    dwell_time_actual:
      data_type: 
      name: 

    # Required for absorption corrections:
    postsample_intensity:
      data_type: 
      name: 

    ########################################################################


    # Optional: location of any extra scalar values to include in the ######
    # nexus representation #################################################
    scalar_data:

    # This is the first item in the list:
    - 
      # Your label for the dataset here. Recommended: use snake_case
      label: 

      # The units in which values were recorded
      units: 

      # How values were recorded in the raw data files. Choices:
      # spec_motor, scan_column, smb_par, or expression (for expressions
      # involving values from one or more of the data streams configured
      # in scalar_data). smb_par is only a valid choice if data for this
      # map was collected at station id1a3 or station id3a.
      data_type:

      # For data_type == spec_motor: the SPEC motor's mnemonic.
      # For data_type == scan_column: the SPEC data column label.
      # For data_type == smb_par: the name of the .par file column (found
      # in the corresponding .json file).
      # For data_type == expression: a mathematical expression that may
      # use the labels of any items in the scalar_data field (found below)
      # and the built-in python function, `round()`. Example usage: an
      # independent dimension of a map is the sum of two spec motors'
      # values. Both spec motors are configured as items in scalar_data
      # for the map, and have labels m1 and m2. Here, name could be:
      # round(m1 + m2, 4)
      name: 

    # This is another item in the list:
    - label: 
      units:
      data_type:
      name:

    ```
    </details>
1. `pyfai_integration_procesor_config.yaml` (configuring a `CHAP.common.models.integration.PyFaiIntegrationProcessorConfig`)

   This configuration contains the instructions for performing one or more kinds of integrations on the raw input dataset.
   Example `pyfai_integration_processor_config.yaml`
   <details>
   <summary>Example `pyfai_integration_processor_config.yaml`</summary>

   ```yaml
   azimuthal_integrators:
   - id: PIL9
     poni_file: LaB6_PIL9_10.30.2024.poni
     mask_file: PIL9_mask.tif
   - id: PIL11
     poni_file: LaB6_PIL11_10.30.2024.poni
     mask_file: PIL11_mask.tif
   integrations:
   - name: waxs_azimuthal
     integration_method: integrate1d
     integration_params:
       npt: 200
       method: bbox_csr_cython
     multi_geometry:
       ais:
       - PIL9
       - PIL11
       unit: q_A^-1
       radial_range:
       - 0.6
       - 4.1
       azimuth_range:
       - -180.0
       - 180.0
   - name: waxs_cake
     integration_method: integrate2d
     integration_params:
       npt_rad: 180
       npt_azim: 360
       method: bbox_csr_cython
     multi_geometry:
       ais:
       - PIL9
       - PIL11
       unit: q_A^-1
       radial_range:
       - 0.6
       - 4.1
       azimuth_range:
       - -180.0
       - 180.0
   - name: waxs_radial
     integration_method: integrate_radial
     integration_params:
       ais:
       - PIL9
       - PIL11
       npt: 360
       npt_rad: 300
       radial_range:
       - 0.6
       - 4.1
       azimuth_range:
       - -180.0
       - 180.0
       unit: chi_deg
       radial_unit: q_A^-1
       method: bbox_csr_cython
   ```
   </details>

   <details>
   <summary>Annotated `pyfai_integration_processor_config.yaml`</summary>

   ```yaml
   # List of detectors whose data will be processed
   azimuthal_integrators:
   - # The detector "indentifier" for filenames, usually the EPICS PV prefix
     id: PIL5
     # A PONI file containing the detetcor calibration data
     poni_file: PIL5_AgBe.poni
     # A file containing a mask to apply to every frame of this detetor's data before processing
     mask_file: PIL5_mask.tif
   - # Another detector may be configured here...

   # List of integrations to perform
   integrations:
   - # A unique name / title for this processed dataset
     name: saxs_azimuthal

     # Choose one of: integrate1d, integrate2d, or integrate_radial
     #   integrate1d -- PyFAI method is: https://pyfai.readthedocs.io/en/stable/api/pyFAI.html#pyFAI.multi_geometry.MultiGeometry.integrate1d
     #   integrate2d -- PyFAI method is: https://pyfai.readthedocs.io/en/stable/api/pyFAI.html#pyFAI.multi_geometry.MultiGeometry.integrate2d
     #   integrate_radial -- PyFAI method is: https://pyfai.readthedocs.io/en/stable/api/pyFAI.html#pyFAI.integrator.azimuthal.AzimuthalIntegrator.integrate_radial
     integration_method: integrate1d

     # Dictionary of initialization arguments for the PyFAI.multi_geometry.MultiGeometry integrator.
     # Required _only_ if `integration_method` is _not_ `integrate_radial`.
     # Go to https://pyfai.readthedocs.io/en/stable/api/pyFAI.html#pyFAI.multi_geometry.MultiGeometry.__init__
     # to see what parameters are accepted.
     multi_geometry:
       ais:
       # List of integrator objects can be passed by just referring to their detectors' "id"s,
       # configured in "azimuthal_integrators".
       - PIL5
       unit: q_A^-1
       radial_range:
       - 0.6
       - 4.1
       azimuth_range:
       - -180.0
       - 180.0
     # Dictionary of parameters for the pyFAI method chosen with `integration_method`
     # Go to the link corresponding to your chosen value of integration_method above
     # to see what parameters your chosen method accepts.
     integration_params:
       npt: 200
       method: bbox_csr_cython
   - # Another integration may be configured here...
   ```
   </details>

### Optimizing performance
Guide on selecting appropriate values for dataset_chunks and running multiple update jobs in parallel

## Notes on corrections calculations
There are currently three convenience tools available for performing corrections: `saxswaxs.FluxCorrectionProcessor`, `saxswaxs.FluxAbsorptionCorrectionProcessor`, and `saxswaxs.FluxAbsorptionBackrgroundCorrectionProcessor`. The exact calculations that each ones performs are detailed below.

### Definitions
- $I_{uncorrected}$ is the uncorrected, integrated dataset. When reading this data as input for a `saxswaxs.*CorrectionProcessor`, use `name: intensity`.
- $I_{incident}$ is the presample intensity dataset. When reading this data as input for a `saxswaxs.*CorrectionProcessor`, use `name: presample_intensity`.
- $I_{transmitted}$ is the postsample intensity dataset. When reading this data as input for a `saxswaxs.*CorrectionProcessor`, use `name: postsample_intensity`.
- $t_{dwell}$ is the scan's dwell time at each point in the map configuration to which the correction tool is applied.  When reading this data as input for a `saxswaxs.*CorrectionProcessor`, use `name: dwell_time_actual`.
- $I_{background}$ is the background integrated detector data. When reading this data as input for a `saxswaxs.*CorrectionProcessor`, use `name: background_intensity`.
- $I_{incident, background}$ is the background presample intensity. When reading this data as input for a `saxswaxs.*CorrectionProcessor`, use `name: background_presample_intensity`.
- $I_{transmitted, background}$ is the background postsample intensity. When reading this data as input for a `saxswaxs.*CorrectionProcessor`, use `name: background_postsample_intensity`.
- $\phi_{reference}$ is `presample_intensity_reference_rate` in the correction tool's own parameters. However, if `presample_intensity_reference_rate` was not given in the tool's configuration, a value for this quantity will be calculated with:

$$
\phi_{reference} = \frac{\overline{I_{incident}}}{\overline{t_{dwell}}}
$$

- $T = \frac{I_{transmitted}}{I_{incident}} / \frac{I_{transmitted, background}}{I_{incident, background}}$
- $t$ is the sample thickness in $cm$. This quantity only appears in `saxswaxs.FluxAbsorptionBackgroundCorrectionProcessor`. The value of this parameter can be set in one of two ways:
  1. Use the `sample_thickness_cm` parameter of `saxswaxs.FluxAbsorptionBackgroundCorrectionProcessor`, OR
  2. Use the `sample_mu_inv_cm` ("$\mu$") parameter of `saxswaxs.FluxAbsorptionBackgroundCorrectionProcessor`. $\mu$ is known as the the linear attenuation coefficient of the sample, and is related to the _mass_ attenuation coefficient, $(\mu/\rho)$ [cm$^2$/g], by the sample density. Tabulated values of the $(\mu/\rho)(E)$ for each element are available here: https://physics.nist.gov/PhysRefData/XrayMassCoef/tab3.html. For our purposes:

$$
t = \frac{-\ln{T}}{\mu}
$$

  - NB: When using `saxswaxs.FluxAbsorptionBackgroundCorrectionProcessor`, do not use both the `sample_thickness_cm` _and_ `sample_mu_inv_cm` parameters at the same time. Specifying both parameters makes the definition of the sample thickness ambiguous. The Processor will raise an error if both parameters are supplied.
- $C_f$ is the scalar factor for putting flux, absorption, background, and thickness corrected data into absolute intensity units. Taken directly from the `absolute_intensity_scalar` parameter from the correction tool config file.

### `saxswaxs.FluxCorrectionProcessor`

$$
I_{corrected} = I_{uncorrected} * \frac{\phi_{reference}}{I_{incident}}
$$

### `saxswaxs.FluxAbsorptionCorrectionProcessor`

$$
I_{corrected} = \frac{1}{T} * I_{uncorrected} * \frac{\phi_{reference}}{I_{incident}}
$$

### `saxswaxs.FluxAbsoprtionBackgroundCorrectionProcessor`

This tool functions differently depending on what parameters are provided:

- If neither `sample_thickness_cm` or `sample_mu_inv_cm` are provided:

$$
I_{corrected} = (I_{uncorrected} * \frac{1}{T} * \frac{\phi_{reference}}{I_{incident}}) - (I_{background} * \frac{\phi_{reference}}{I_{incident, background}})
$$

- If either `sample_thickness_cm` or `sample_mu_inv_cm` are provided:

$$
I_{corrected} = \frac{1}{t} * [(I_{uncorrected} * \frac{1}{T} * \frac{\phi_{reference}}{I_{incident}}) - (I_{background} * \frac{\phi_{reference}}{I_{incident, background}})]
$$




## Running a workflow

### At CHESS
Use environment: `source /nfs/chess/sw/miniforge3_chap/bin/activate; conda activate CHAP_saxswaxs`
And/or add this to your ~/.bashrc: `alias CHAP_saxswaxs='/nfs/chess/sw/miniforge3_chap/envs/CHAP_saxswaxs/bin/CHAP'`

### Batch jobs
Drop in & replace: PyfaiIntegrationProcessorConfig
idx_slice on update jobs will need adjustments to drop in & replace: MapConfig
Script for constructuing and / or sending off multiple update jobs in parallel
