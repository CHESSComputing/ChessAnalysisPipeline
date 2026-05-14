(chap_pipeline)=
# CHAP Pipeline

To run a CHESS Analysis Pipeline (`CHAP`), you will need:

1. A `CHAP` configuration file in YAML format
2. A `CHAP` command line executable (CLI) executable

Run a `CHAP` pipeline by executing:

```bash
$ CHAP pipeline.yaml
```

How to run `CHAP` on the CHESS Linux system with centrally maintained workflow executables is discussed [below](chap_executables_chess).

## Constructing a `CHAP` configuration file
`CHAP` configuration files must be in [YAML format](https://en.wikipedia.org/wiki/YAML). At the top level, the file contains a single document, the document contains a single structure, the structure contains at least two keys, one of the keys must be `config`, and all other keys are pipeline names.

Example of a complete `CHAP` pipeline configuration file:
```yaml
config:
  root: .
pipeline:
- common.YAMLReader:
    filename: data.yaml
- common.PrintProcessor
```

### The `config` section
The `config` section contains the values of the instance variables for an instance of [`CHAP.models.RunConfig`](https://chesscomputing.github.io/ChessAnalysisPipeline/CHAP.html#CHAP.models.RunConfig). It is techinically optional, but it should be included in every pipeline file for reproducibility / provenance. It can also be helpful for applying the same pipeline on many datasets, depending on how your dataset files are organized. The keys you can use in this section are:

| Key | Description | Default value |
| --- | ----------- | ------------- |
| `root`| Path to the working directory | Directory from which<br>`CHAP` was run |
| `inputdir` | Path to a directory where all `Reader`s will look for files | Same value as `root` |
| `outputdir` | Path to a directory where all `Writer`s will write files | Same value as `root` |
| `interactive` | Flag to allow certain optional data / parameter checks that<br>require user interaction to proceed. Not applicable to all pipelines,<br>only to those which contain these optionally-interactive tools. | `false`
| `log_level` | Name of a python logging level (not case sensitive) | `info`


Example `config` section containing all default values:
```yaml
config:
  root: .
  inputdir: .
  outputdir: .
  interactive: false
  log_level: info
```

### Pipeline sections
Sections with names that are not `config` are actual pipelines. A single CHAP configuration file may contain more than one pipeline. Each pipeline must be an list of `Reader`s, `Processor`s, and `Writer`s (`Pipelinetem`s) to execute consecutively, and configure the instance variables and other parameters for each one. To assemble your own pipeline configuration:

1. Decide which `PipelineItem`s to use and in what order.
1. For each `PipelineItem`, refer to the [Reference Guide (API documentation)](api_documentation.rst) to find out what *instance variables* it has. The Reference Guide also contain a description of every variable, its expected type, and its default value (for optional variables). Remember to include the instance variables for any object from which the relevant `PipelineItem` inherrits. For example, [`YAMLReader`](https://chesscomputing.github.io/ChessAnalysisPipeline/CHAP.common.html#CHAP.common.reader.YAMLReader) lists no instance variables, but it does inherit from [`Reader`](https://chesscomputing.github.io/ChessAnalysisPipeline/CHAP.html#CHAP.reader.Reader), which has [`filename`](https://chesscomputing.github.io/ChessAnalysisPipeline/CHAP.html#CHAP.reader.Reader.filename), so `YAMLReader` also has the `filename` instance variable.

#### Example: `MapProcessor`
Suppose you want to configure a pipeline that collects all raw data from a CHESS dataset in a [NeXus](https://www.nexusformat.org) file, and that you already have a valid [`CHAP.common.models.map.MapConfig`](https://chesscomputing.github.io/ChessAnalysisPipeline/CHAP.common.models.html#CHAP.common.models.map.MapConfig) object for the dataset saved to a file named `map_config.yaml`. To create a suitable pipeline file:

1. Decide on the required `PipelineItem`s. The pipeline will need a `Reader` that supports YAML files, a `Processor` that collects `MapConfig` data in a NeXus structure, and a `Writer` that supports NeXus files. So, the pipeline configuration looks like this to start:
   ```yaml
   pipeline:
   - common.YAMLReader:
     TBD
   - common.MapProcessor:
     TBD
   - common.NexusWriter:
     TBD
   ```
2. Now, fill in all the TBD's by referring to the Reference Guide for each `PipelineItem` to specify the instance variables.
   ```yaml
   pipeline:
   - common.YAMLReader:
     filename: map_config.yaml
   - common.MapProcessor:
     detector_config:
       detectors:
       - id: detector_id
         shape: [0, 0]
         attrs:
           foo: bar
   - common.NexusWriter:
     filename: map_data.nxs
   ```

## `CHAP` CLI usage
To diplay a description on how to use `CHAP` from the command line, execute:
```
$ CHAP --help
```
to get:
```
usage: PROG [-h] [-p [PIPELINE ...]] [--regex [{match,search,fullmatch}]]
            [--batch] [--batch-logdir LOGDIR]
            config

positional arguments:
  config                Input configuration file

options:
  -h, --help            show this help message and exit
  -p [PIPELINE ...], --pipeline [PIPELINE ...]
                        Pipeline name(s)
  --regex [{match,search,fullmatch}]
                        Name of Python RegEx function
                        (https://docs.python.org/3/howto/regex.html) to use
                        for matching configured pipeline names against the
                        string provided with the -p / --pipeline option.
  --batch               Enables "batch mode" operation where every sub-
                        pipeline is run in separate parallel processes. Log
                        files for each pipeline process will be created in the
                        directory specified with the `--batch-logdir` option.
  --batch-logdir LOGDIR
                        Destination directory for individual pipeline log
                        files when running multiple pipelines in batch mode.
```
| Option | Description |
| ------ | ----------- |
| `--pipeline` or `-p` | When more than one named pipeline configuration is present in a `CHAP` config file,<br>`--pipeline` or `-p` can be used to specify a limited selection of the pipeline(s) from<br>the file to be executed. |
| `--regex` | This option augments the behavior of `--pipeline` -- the difference is that when<br>`--regex` is used, the value of `--pipeline` specfies a [regular expression](https://docs.python.org/3/howto/regex.html) pattern for<br>selecting the names of pipeline(s) to run. |
| `--batch` | This option augments the behavior of `--pipeline` -- the difference is that when `--batch` is used, the specified pipelines are executed individually and in parallel instead of<br>being concatenated and executed as a single pipeline. |

### Example commands
Suppose `pipeline.yaml` contains:
```yaml
config:
  root: .
pipeline_1:
- common.YAMLReader:
    filename: data_1.yaml
- common.PrintProcessor
pipeline_2:
- common.YAMLReader:
    filename: data_2.yaml
- common.PrintProcessor
```
| Command | Behavior |
| ------- | -------- |
| `CHAP pipeline.yaml` or<br>`CHAP pipeline.yaml --regex -p pipeline --regex` | Concatenate `pipeline_1` and `pipeline_2` and execute all items as a single pipeline. |
| `CHAP pipeline.yaml -p pipeline_1` | Execute `pipeline_1` only. |
| `CHAP pipeline.yaml --batch` or<br>`CHAP pipeline.yaml -p pipeline --regex --batch`| Execute `pipeline_1` and `pipeline_2` in separate parallel processes, creating a log file for each one: `./CHAP_logs/pipeline_1.log` and `./CHAP_logs/pipeline_2.log`. |

(chap_executables_chess)=
## Python executables for `CHAP` on the CHESS Linux system

Running `CHAP` on the CHESS Linux system does not require users to create their own Conda environment or `CHAP` executables. Instead CHESS maintains regularly updated `CHAP` executables to run any of the maintained workflows located in the shared software releases directory for CHESS: `/nfs/chess/sw/CHESS-software-releases`. Specifically, production and development versions of the `CHAP` executables can be found in `/nfs/chess/sw/CHESS-software-releases/prod` and `/nfs/chess/sw/CHESS-software-releases/dev`, respectively.

Production version executables are updated each time a new tagged release is created for the main branch of the [CHAP Github repository](https://github.com/CHESSComputing/ChessAnalysisPipeline). Links to executables for the latest production version can be found in `/nfs/chess/sw/CHESS-software-releases/prod`, links to older releases can be found in subdirectories identified by its release version number. Release notes can be found [here](https://github.com/CHESSComputing/ChessAnalysisPipeline/releases). The `CHAP` [Reference Guide (API documentation)](api_documentation.rst) is also updated automatically with each new tagged release.

Development version executables are updated each time a new commit is pushed to the [`dev` branch](https://github.com/CHESSComputing/ChessAnalysisPipeline/tree/dev) of the [CHAP Github repository](https://github.com/CHESSComputing/ChessAnalysisPipeline). Links to executables for the latest development version can be found in `/nfs/chess/sw/CHESS-software-releases/dev`.

For example, to run the [Tomo workflow](tomo_workflow) using the latest production release version, execute:

```bash
$ /nfs/chess/sw/CHESS-software-releases/prod/CHAP_tomo pipeline.yaml
```

or to run the [EDD workflow](edd_workflow) using the latest development release version, execute:

```bash
$ /nfs/chess/sw/CHESS-software-releases/dev/CHAP_edd pipeline.yaml
```

You may find it convenient to add an alias to your `~/.bascrc` or `~/.bash_aliases`, for example for the `CHAP` Tomography workflow production release:
```bash
alias CHAP_tomo_prod='/nfs/chess/sw/CHESS-software-releases/prod/CHAP_tomo'
```
after which you can run the [Tomo workflow](tomo_workflow) using the latest production release version by simply executing:

```bash
$ CHAP_tomo_prod pipeline.yaml
```

(build_chap_environment)=
## Python environments for `CHAP` on any Linux system

Developing a user `PipelineItem` for `CHAP` or running `CHAP` on a Linux system other than the CHESS farm does require users to create their own Conda environment by taking the following steps:

1. Create a base Conda environent and clone the `CHAP` repository according to steps 1 and 2 of the {ref}`Conda installation instructions <conda_installation>`.
1. Create a Conda environment suitable to your own `PipelineItem` or create a Conda environment for each workflow that you want to run.

For example, to create the SAXSWAXS Conda environment and run a SAXSWAXS workflow:
1. Activate your base Conda environment:
   ```bash
   $ source <path_to_CHAP_clone_dir>/bin/activate
   ```
1. Create a Conda environment inside your base environment with:
   ```bash
   (base) $ mamba env create -f <path_to_CHAP_clone_dir>/CHAP/saxswaxs/environment.yml
   ```
1. Activate the `CHAP_saxswaxs` environment:
   ```bash
   (base) $ conda activate CHAP_saxswaxs
   ```
1. Try running:
   ```bash
   (CHAP_saxswaxs) $ CHAP --help
   ```
   to confirm that the package and the environment were installed correctly.
1. Navigate to your work directory.
1. Create the required `CHAP` pipeline file for the workflow (see above) and any additional workflow specific input files.
1. Run the workflow using your own `CHAP_saxswaxs` executable:
```bash
   (CHAP_saxswaxs) $ CHAP pipeline.yaml
   ```
