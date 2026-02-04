# SAXS/WAXS module (`CHAP.saxswaxs`)

The `CHAP.saxswaxs` module contains processing tools unique to SAXS/WAXS processing. This document describes how to use them in a pipeline configuration YAML file so that SAXS/WAXS workflows can be run from the terminal on the CLASSE Linux system.


## Workflow description

### Overview of steps

#. Setup container for the complete dataset
   First, we set up a container to hold the raw and processed datasets before any processing begins. This step needs the following information to allocate an appropriate amount of space:
   #. Dataset size
   #. Detector size(s)
   #. Integrated data size(s)
   This step also sets the number of chunks for each data array in the container. [Selecting the right number of chunks is important for optimizing performance](#Optimizing_performance) during the next step.

#. Fill container with processed data
   Next, we read in the raw data and perform the configured integration(s). To get the best performance for large datasets, this step should be performed across more than one process runnning in parallel. Each process will handle reading, processing, and writing just one part of the whole dataset. Each one of the parallel jobs should fill out exactly one chunk of each array in the container set up previously. To fill the container from the previous step with processed data, the following information will be required:
   #. Specifications of the slice of arrays in the container to which results should be written
- update_*

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