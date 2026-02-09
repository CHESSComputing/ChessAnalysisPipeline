# SAXS/WAXS module (`CHAP.saxswaxs`)

The `CHAP.saxswaxs` module contains processing tools unique to SAXS/WAXS processing. This document describes how to use them in a pipeline configuration YAML file so that SAXS/WAXS workflows can be run from the terminal on the CLASSE Linux system.


## Workflow description

### General overview

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
Take from saxswaxsworkflow wiki


## Configuring a pipeline

### PyfaiIntegrationProcessorConfig examples

### Pipeline examples


## Running a workflow

### At CLASSE
Use environment: `source /nfs/chess/sw/miniforge3_chap/bin/activate; conda activate CHAP_saxswaxs`

### Batch jobs
Drop in & replace: PyfaiIntegrationProcessorConfig
idx_slice on update jobs will need adjustments to drop in & replace: MapConfig
Script for constructuing and / or sending off multiple update jobs in parallel