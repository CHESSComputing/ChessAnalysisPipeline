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

#. Perform final user adjustements
- convert
  - describe data.zarr, AND what's missing
- correct
  - setup background
  - perform corrections
- makelink
  - axes are important
- struct
  - visualization

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