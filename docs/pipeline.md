(chap_pipeline)=
# CHAP Pipeline

To run a pipeline, you will need:

1. A `CHAP` configuration file in YAML format
2. A `CHAP` CLI executable

Run a pipeline by executing:

```bash
$ CHAP pipeline.yaml
```

## Constructing a `CHAP` configuration file
CHAP configuration files must be in [YAML format](https://en.wikipedia.org/wiki/YAML). At the top level, the file contains a single document, the document contains a single structure, the structure contains at least two keys, one of the keys must be `config`, and all other keys are pipeline names.

Example of a complete CHAP pipeline configuration file:
```yaml
config:
  root: .
pipeline:
- common.YAMLReader:
    filename: data.yaml
- common.PrintProcessor
```

### The `config` section
This section contains the values of the instance variables for an instance of [`CHAP.models.RunConfig`](https://chesscomputing.github.io/ChessAnalysisPipeline/CHAP.html#CHAP.models.RunConfig). It is techinically optional, but it should be included in every pipeline file for reproducibility / provenance. It can also be helpful for applying the same pipeline on many datasets, depending on how your dataset files are organized. The keys you can use in this section are:

| Key | Description | Default value |
| --- | ----------- | ------------- |
| `root`| Path to the working directory | Working directory from which `CHAP` was run |
| `inputdir` | Path to a directory where all `Reader`s will look for files | Same value as `root` |
| `outputdir` | Path to a directory where all `Writer`s will write files | Same value as `root` |
| `interactive` | Flag to allow certain optional data / parameter checks that require user interaction to proceed. Not applicable to all pipelines, only to those which contain these optionally-interactive tools. | `false`
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
Sections with names that are not `config` are actual pipelines. A single CHAP configuration file may contain more than one pipeline. Each pipeline must be an list of `Reader`s, `Processor`s, and `Writer`s (`Pipelinetem`s) to execute in order, and configure the instance variables and other parameters for each one. To assemble your own pipeline configuration:

1. Decide which `PipelineItem`s to use and in what order
1. For each `PipelineItem`, refer to the API documentation to find out what *instance variables* it has. The docs also contain a description of every variable, its expected type, and its default value (if the variable is optional). Remember to include the instance variables for any object from which the relevant `PipelineItem` inhertits. For example, [`YAMLReader`](https://chesscomputing.github.io/ChessAnalysisPipeline/CHAP.common.html#CHAP.common.reader.YAMLReader) lists no instance variables, but it does inherit from [`Reader`](https://chesscomputing.github.io/ChessAnalysisPipeline/CHAP.html#CHAP.reader.Reader), which has [`filename`](https://chesscomputing.github.io/ChessAnalysisPipeline/CHAP.html#CHAP.reader.Reader.filename), so `YAMLReader` also has `filename`. 

#### Example: `MapProcessor`
Suppose you want to configure a pipeline that collects all raw data from a CHESS dataset in a NeXus file, and that you already have a valid [`CHAP.common.models.map.MapConfig`](https://chesscomputing.github.io/ChessAnalysisPipeline/CHAP.common.models.html#CHAP.common.models.map.MapConfig) object for the dataset saved to a file named `map_config.yaml`

1. Decide on the `PipelineItem`s. It'll need a `Reader` that supports YAML files, a `Processor` that collects `MapConfig` data in a NeXus structure, and a `Writer` that supports Nexus files. So, the pipeline configuration looks like this to start:
   ```yaml
   pipeline:
   - common.YAMLReader:
     # TBD
   - common.MapProcessor:
     # TBD
   - common.NexusWriter:
     # TBD
   ```
2. Now, fill in all the `# TBD` by referring to the variables for each item.
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
```
$ CHAP --help
usage: PROG [-h] [-p [PIPELINE ...]] [--regex [{match,search,fullmatch}]] [--batch] [--batch-logdir LOGDIR] config

positional arguments:
  config                Input configuration file

options:
  -h, --help            show this help message and exit
  -p [PIPELINE ...], --pipeline [PIPELINE ...]
                        Pipeline name(s)
  --regex [{match,search,fullmatch}]
                        Name of Python regex function to use for matching configured pipeline names against the string provided with the -p / --pipeline option.
  --batch               Enables "batch mode" where every sub-pipeline is run in a separate process, all at once. Log files for each pipeline process will be created in the directory
                        specified with the `--batch-logdir` option.
  --batch-logdir LOGDIR
                        Destination directory for individual pipeline log files when running multiple pipelines in batch mode.
```
| Option | Description |
| ------ | ----------- |
| `-p` or `--pipeline` | When more than one named pipeline configuration is present in a CHAP config file, `--pipeline` or `-p` can be used to specify a limited selection of the pipeline(s) from the file to be executed. |
| `--regex` | This option augments the behavior of `--pipeline` -- the difference is that when `--regex` is used, the value of `--pipeline` specfies a *regular expression pattern* for selecting the names of pipeline(s) to run. |
| `--batch` | This option augments the behavior of `--pipeline` -- the difference is that when `--batch` is used, the specified pipelines are executed individually and in parallel instead of being concatenated and executed as a single pipeline. |

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
| `CHAP pipeline.yaml` or<br>`CHAP pipeline.yaml --regex -p pipeline --regex` | Concatenate `pipeline_1` and `pipeline_2` and execute all items as a single pipeline |
| `CHAP pipeline.yaml -p pipeline_1` | Execute `pipeline_1` only |
| `CHAP pipeline.yaml --batch` or<br>`CHAP pipeline.yaml -p pipeline --regex --batch`| Execute `pipeline_1` and `pipeline_2` in separate parallel processes, creating a log file for each one: `./CHAP_logs/pipeline_1.log` and `./CHAP_logs/pipeline_2.log`

## Python environments for `CHAP` on the CLASSE Linux system
TODO
