(introduction_ref)=
# Introduction

The CHESS Analysis Pipeline (CHAP), depeloped at the [Cornell High Energy Synchrotron Source](https://www.chess.cornell.edu) (CHESS),  is an object-oriented python framework for refactoring monolithic data
analysis programs into modular pipelines composed of interchangeable, reusable code components. The basic
blueprint for a `Pipeline` component consists of a `Reader`, `Processor`, and `Writer`, as shown in the diagram below:

```{figure} diagrams/chap-base.png
---
figclass: center-img-only
name: schematic_chap_pipeline_component
---
The basic blueprint of a CHAP pipeline component.
```

The `Reader` and `Writer` base classes handle data input and output for a `Pipeline` component, respectively. These base classes encapsulate the CHESS-specific logistics, such as file operations and data format conversions and validate their data with the [Pydantic library](https://github.com/pydantic/pydantic). They isolate a `Pipeline`'s I/O functions from its data analysis algorithms, allowing a single algorithm to accept multiple data formats.
Inherited subclasses are defined for specific file types, e.g. `H5Reader`, `NexusWriter`. A single `Pipeline` component may contain multiple `Reader`s or `Writer`s.

The `Processor` base class encapsulates the data analysis algorithm. Because `Processor`s are independent of
CHESS infrastructure, researchers can easily contribute `Processor` subclasses
with bespoke data analysis code. `Reader`s and `Writer`s pass data into and out of `Processor`s via `PipelineData` containers.
Multiple `Processor`s can be chained together within a single `Pipeline` component.

Workflows are defined by CHAP configuration files written in YAML. Each file may contain one or more
`Pipeline` components that can be executed individually or all at once (sequentially or in parallel). The  diagram below shows
a schematic workflow with series of `Pipeline` components linked together in a single configuration file.

```{figure} diagrams/chap-schematic.png
---
figclass: center-img-only
name: schematic_chap_workflow
---
An example of a CHAP workflow consisting of three `Pipeline` components.
```

Workflows for specific X-ray techniques are constructed from concrete implementations of the CHAP base classes.
For example, in the [EDD workflow example pipeline file](edd_pipeline),
"energy" is for energy calibration with "Processor 0" being `edd.MCAEnergyCalibrationProcessor`,
"twotheta" is for detector theta calibration with "Processor 1" being `edd.MCATthCalibrationProcessor`, and so on.
