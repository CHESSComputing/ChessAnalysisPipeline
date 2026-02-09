# SAXS/WAXS module (`CHAP.saxswaxs`)

The `CHAP.saxswaxs` module contains processing tools unique to SAXS/WAXS processing. This document describes how to use them in a pipeline configuration YAML file so that SAXS/WAXS workflows can be run from the terminal on the CLASSE Linux system.


## Workflow description

### General overview

#### Setup
#. Setup a container for the complete raw and processed dataset
   First, we set up a container to hold the raw and processed datasets before any processing begins. This step needs the following information to allocate an appropriate amount of space:
   #. Dataset size
   #. Detector size(s)
   #. Integrated data size(s)
   In practice, you should prepare [two supplementary configuration files](#related_configuration_objects) in addition to the parameters for the tools involved in this step: one for a `MapConfig`, object, another for a `PyFaiIntegrationProcessorConfig` object. 
   This step also sets the number of chunks for each data array in the container. [Selecting the right number of chunks is important for optimizing performance](#Optimizing_performance) during the next step.

   Example pipeline configuration:
   ```yaml
   config:
     root: .
     log_level: debug
   setup:
   # Tool 1: read in configuration files
   - pipeline.MultiplePipelineItem:
       items:
       - common.YAMLReader:
	   filename: map_config.yaml
	   schema: common.models.map.MapConfig
       - common.YAMLReader:
	   filename: pyfai_integration_processor_config.yaml
	   schema: common.models.integration.PyfaiIntegrationConfig
   # Tool 2: set up Zarr container for processed data based on
   #         configurations provided from Tool 1
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
   # Tool 3: Write the Zarr container to file
   - common.ZarrWriter:
       filename: data.zarr
       force_overwrite: true   
   ```
   
#### Update
#. Fill container with processed data
   Next, we read in the raw data and perform the configured integration(s). To get the best performance for large datasets, this step should be performed across multiple pipeline processes runnning in parallel. Each process will handle reading, processing, and writing just one part of the whole dataset -- exactly _one chunk_ of each array in the container set up previously, to be precise. To fill the container from the previous step with processed data, each parallel process will need to know the following information:
   #. The spec file for the scan whose data will be processed
   #. The scan number for the scan whose data will be processed
   #. Parameters indicating a specific slice of the scan to process

   Example pipeline:
   ```yaml
   config:
     root: .
     log_level: debug
   update:
   # Tool 1: read in configuration files
   - pipeline.MultiplePipelineItem:
       items:
       - common.YAMLReader:
	   filename: map_config.yaml
	   schema: common.models.map.MapConfig
       - common.YAMLReader:
	   filename: pyfai_integration_processor_config.yaml
	   schema: common.models.integration.PyfaiIntegrationConfig
   # Tool 2: read and process the data
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
   # Tool 3: write processed data to the container set up earlier
   - common.ZarrValuesWriter:
       filename: data.zarr
   ```

#### Finalize
#. Perform final user adjustments
   These "final adjustments" are all optional, but they can include the following:
   - converting from .zarr format to NeXus file format
   - performing flux, absorption, and / or background corrections
   - inserting links to coordinate axes next to processed data arrays
   - reshaping the dataset from an "unstructured" to a "structured" representation

   Examples:

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

   ```yaml
   config:
     root: .
     log_level: debug
   flux_correct:
   # Tool 1: Read in uncorrected data and flux data to perform flux correction
   - pipeline.MultiplePipelineItem:
       items:
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
   # Tool 2: Perform flux correction calculations
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

   ```yaml
   config:
     root: .
     log_level: debug
   linkdims:
   # Tool 1: Read in the Nexus file in which we're adding linked data fields
   - common.NexusReader:
       filename: data.nxs
       mode: rw
   # Tool 2: Create new linked data arrays from all the groups listed
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

   ```yaml
   config:
     root: .
     log_level: debug
   struct:
   # Tool 1: Read in the data arrays to structure, and all their
   # coordinate axes arrays too
   - pipeline.MultiplePipelineItem:
       items:
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
   # Tool 2: restructure signal arrays in a single shared NXdata object
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
   # Tool 3: Write the structured data to a new group in the original NeXus file
   - common.NexusWriter:
       filename: data.nxs
       nxpath: /structured
       force_overwrite: true
   ```

### Optimizing performance
Guide on selecting appropriate values for dataset_chunks and running multiple update jobs in parallel

### Data in / output formats
CHESS data in, .zarr data out

## Related Configuration Objects
### MapConfig
#### Description
#### Example
### PyFaiIntegrationProcessorConfig
#### Description
#### Example

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
```math
\phi_{reference} = \frac{\overline{I_{incident}}}{\overline{t_{dwell}}}
```
- $T = \frac{I_{transmitted}}{I_{incident}} / \frac{I_{transmitted, background}}{I_{incident, background}}$
- $t$ is the sample thickness in $cm$. This quantity only appears in `saxswaxs.FluxAbsorptionBackgroundCorrectionProcessor`. The value of this parameter can be set in one of two ways:
  1. Use the `sample_thickness_cm` parameter of `saxswaxs.FluxAbsorptionBackgroundCorrectionProcessor`, OR
  2. Use the `sample_mu_inv_cm` ("$\mu$") parameter of `saxswaxs.FluxAbsorptionBackgroundCorrectionProcessor`. $\mu$ is known as the the linear attenuation coefficient of the sample, and is related to the _mass_ attenuation coefficient, $(\mu/\rho)$ [cm$^2$/g], by the sample density. Tabulated values of the $(\mu/\rho)(E)$ for each element are available here: https://physics.nist.gov/PhysRefData/XrayMassCoef/tab3.html. For our purposes:
    ```math
    t = \frac{-\ln{T}}{\mu}
    ```
  - NB: When using `saxswaxs.FluxAbsorptionBackgroundCorrectionProcessor`, do not use both the `sample_thickness_cm` _and_ `sample_mu_inv_cm` parameters at the same time. Specifying both parameters makes the definition of the sample thickness ambiguous. The Processor will raise an error if both parameters are supplied.
- $C_f$ is the scalar factor for putting flux, absorption, background, and thickness corrected data into absolute intensity units. Taken directly from the `absolute_intensity_scalar` parameter from the correction tool config file.

### `saxswaxs.FluxCorrectionProcessor`
```math
I_{corrected} = I_{uncorrected} * \frac{\phi_{reference}}{I_{incident}}
```
### `saxswaxs.FluxAbsorptionCorrectionProcessor`
```math
I_{corrected} = \frac{1}{T} * I_{uncorrected} * \frac{\phi_{reference}}{I_{incident}}
```
### `saxswaxs.FluxAbsoprtionBackgroundCorrectionProcessor`

This tool functions differently depending on what parameters are provided:

- If neither `sample_thickness_cm` or `sample_mu_inv_cm` are provided:
```math
I_{corrected} = (I_{uncorrected} * \frac{1}{T} * \frac{\phi_{reference}}{I_{incident}}) - (I_{background} * \frac{\phi_{reference}}{I_{incident, background}})
```
- If either `sample_thickness_cm` or `sample_mu_inv_cm` are provided:
```math
I_{corrected} = \frac{1}{t} * [(I_{uncorrected} * \frac{1}{T} * \frac{\phi_{reference}}{I_{incident}}) - (I_{background} * \frac{\phi_{reference}}{I_{incident, background}})]
```

## Configuring a pipeline
Before constructing a `CHAP` pipeline configuration to run a complete SAXS/WAXS data processing workflow, users should first assemble two supplementary configuration files. *Both* these configurations will need to be read in to the pipeline for [the setup step](#setup) and every [update step](#update).
#. `map_config.yaml` (configuraing a `CHAP.common.models.map.MapConfig`)
   This configuration contains everything `CHAP` needs to know about the location, format, and size of the raw input dataset.
   Example `map_config.yaml`:
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
#. `pyfai_integration_procesor_config.yaml` (configuring a `CHAP.common.models.integration.PyFaiIntegrationProcessorConfig`)
   This configuration contains the instructions for performing one or more kinds of integrations on the raw input dataset.
   Example `pyfai_integration_processor_config.yaml`:
   ```yaml
   azimuthal_integrators:
   - id: PIL9
     poni_file: /nfs/chess/aux/reduced_data/cycles/2024-3/id3b/koerner-4187-a/setup/LaB6_PIL9_10.30.2024.poni
     mask_file: /nfs/chess/aux/reduced_data/cycles/2024-3/id3b/koerner-4187-a/setup/PIL9_mask.tif
   - id: PIL11
     poni_file: /nfs/chess/aux/reduced_data/cycles/2024-3/id3b/koerner-4187-a/setup/LaB6_PIL11_10.30.2024.poni
     mask_file: /nfs/chess/aux/reduced_data/cycles/2024-3/id3b/koerner-4187-a/setup/PIL11_mask.tif
   integrations:
   - name: waxs_azimuthal
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
     integration_method: integrate1d
     integration_params:
       npt: 200
       method: bbox_csr_cython
   - name: waxs_cake
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
     integration_method: integrate2d
     integration_params:
       npt_rad: 180
       npt_azim: 360
       method: bbox_csr_cython
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


## Running a workflow

### At CLASSE
Use environment: `source /nfs/chess/sw/miniforge3_chap/bin/activate; conda activate CHAP_saxswaxs`
And/or add this to your ~/.bashrc: `alias CHAP_saxswaxs='/nfs/chess/sw/miniforge3_chap/envs/CHAP_saxswaxs/bin/CHAP'`

### Batch jobs
Drop in & replace: PyfaiIntegrationProcessorConfig
idx_slice on update jobs will need adjustments to drop in & replace: MapConfig
Script for constructuing and / or sending off multiple update jobs in parallel