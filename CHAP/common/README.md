# CHAP.common
## reader.py
Contains implementations for subclasses of `CHAP.Reader` that are (or can be) used by multiple experiment-specific pipelines.
## processor.py
Contains implementations for subclasses of `CHAP.Processor` that are (or can be) used by multiple experiment-specific pipelines.
## writer.py
Contains implementations for subclasses of `CHAP.Writer` that are (or can be) used by multiple experiment-specific pipelines.
## models/
Contains definitions of [`pydantic`](https://docs.pydantic.dev)-based objects to assist in validating configuration data typically passed to `Processor`s that are (or can be) used by multiple experiment-specific pipelines.