### Pipeline
This package conains proof of concepts pipeline framework for workflow
execution. It requires proper configuration of pipeline in terms classes, e.g.
```
# pipeline deifinition as sequence of objects
pipeline:
  - reader.Reader
  - processor.Processor
  - fitter.Fitter
  - processor.Processor
  - writer.Writer
  - fitter.Fitter
  - writer.Writer

# specific object parameters, e.g. our reader accepts fileName=data.csv
reader.Reader:
  fileName: data.csv


# specific object parameters, e.g. our writer accepts fileName=data.out
writer.Writer:
  fileName: data.out
```

Then, you may execute this pipeline as following:
```
./runner.py --config config.yaml
```
and, check the output in `data.out` file.

