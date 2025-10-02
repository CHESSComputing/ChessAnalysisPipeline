"""
File       : runner.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description:
"""

# System modules
from argparse import ArgumentParser
import logging
import os
from typing import (
    Literal,
    Optional,
)
from yaml import safe_load

# Third party modules
from pydantic import (
    DirectoryPath,
    conint,
    field_validator,
    model_validator,
)

# Local modules
from CHAP.models import CHAPBaseModel
from CHAP.pipeline import Pipeline


class RunConfig(CHAPBaseModel):
    """Pipeline run configuration class.

    :ivar root: Default work directory, defaults to the current run
        directory.
    :type root: str, optional
    :ivar inputdir: Input directory, used only if any input file in the
        pipeline is not an absolute path, defaults to `'root'`.
    :type inputdir: str, optional
    :ivar outputdir: Output directory, used only if any output file in
        the pipeline is not an absolute path, defaults to `'root'`.
    :type outputdir: str, optional
    :ivar interactive: Allows for user interactions,
        defaults to `False`.
    :type interactive: bool, optional
    :ivar log_level: Logger level (not case sensitive),
        defaults to `'INFO'`.
    :type log_level: Literal[
        'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], optional
    :ivar profile: Allows for code profiling, defaults to `False`.
    :type profile: bool, optional
    :ivar spawn: Internal use only, flag to check if the pipeline is
        executed as a worker spawned by another Processor.
    :type spawn: int, optional
    """
    root: Optional[DirectoryPath] = None
    inputdir: Optional[DirectoryPath] = None
    outputdir: Optional[DirectoryPath] = None
    interactive: Optional[bool] = False
    log_level: Optional[Literal[
        'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']] = 'INFO'
    profile: Optional[bool] = False
    spawn: Optional[int] = 0

    @model_validator(mode='before')
    @classmethod
    def validate_config(cls, data):
        """Ensure that valid directory paths are provided.

        :param data: Pydantic validator data object.
        :type data: RunConfig,
            pydantic_core._pydantic_core.ValidationInfo
        :return: The currently validated list of class properties.
        :rtype: dict
        """
        if isinstance(data, dict):
            # System modules
            from tempfile import NamedTemporaryFile

            # Make sure os.makedirs is only called from the root node
            comm = data.get('comm')
            if comm is None:
                rank = 0
            else:
                rank = comm.Get_rank()

            # Check if root exists (create it if not) and is readable
            root = data.get('root', os.getcwd())
            if not rank:
                if not os.path.isdir(root):
                    os.makedirs(root)
                if not os.access(root, os.R_OK):
                    raise OSError('root directory is not accessible for '
                                  f'reading ({root})')
            data['root'] = root

            # Check if inputdir exists and is readable
            inputdir = data.get('inputdir', '.')
            if not os.path.isabs(inputdir):
                inputdir = os.path.normpath(os.path.realpath(
                    os.path.join(root, inputdir)))
            if not rank:
                if not os.path.isdir(inputdir):
                    raise OSError(
                        f'input directory does not exist ({inputdir})')
                if not os.access(inputdir, os.R_OK):
                    raise OSError(
                        'input directory is not accessible for reading '
                        f'({inputdir})')
            data['inputdir'] = inputdir

            # Check if outputdir exists (create it if not) and is writable
            outputdir = data.get('outputdir', '.')
            if not os.path.isabs(outputdir):
                outputdir = os.path.normpath(os.path.realpath(
                    os.path.join(root, outputdir)))
            if not rank:
                if not os.path.isdir(outputdir):
                    os.makedirs(outputdir)
                try:
                    NamedTemporaryFile(dir=outputdir)
                except:
                    raise OSError('output directory is not accessible for '
                                  f'writing ({self.outputdir})')
            data['outputdir'] = outputdir

            # Make sure os.makedirs completes before continuing
            # Make sure barrier() is also called on the main node if
            # this is called from a spawned slave node
            if comm is not None:
                comm.barrier()

        return data

    @field_validator('log_level', mode='before')
    @classmethod
    def validate_log_level(cls, log_level):
        """Capitalize log_level."""
        return log_level.upper()


def parser():
    """Return an argument parser for the `CHAP` CLI. This parser has
    one argument: the input CHAP configuration file.
    """
    pparser = ArgumentParser(prog='PROG')
    pparser.add_argument(
        'config', action='store', default='', help='Input configuration file')
    pparser.add_argument(
        '-p', '--pipeline', nargs='*', help='Pipeline name(s)')
    return pparser

def main():
    """Main function."""
    try:
        # Third party modules
        from mpi4py import MPI

        have_mpi = True
        comm = MPI.COMM_WORLD
    except ImportError:
        have_mpi = False
        comm = None

    args = parser().parse_args()

    # Read the input config file
    configfile = args.config
    with open(configfile) as file:
        config = safe_load(file)

    # Check if executed as a worker spawned by another Processor
    run_config = RunConfig(**config.pop('config'), comm=comm)
    if have_mpi and run_config.spawn:
        sub_comm = MPI.Comm.Get_parent()
        common_comm = sub_comm.Merge(True)
        # Read worker specific input config file
        if run_config.spawn > 0:
            with open(f'{configfile}_{common_comm.Get_rank()}') as file:
                config = safe_load(file)
                run_config = RunConfig(
                    **config.pop('config'), comm=common_comm)
        else:
            with open(f'{configfile}_{sub_comm.Get_rank()}') as file:
                config = safe_load(file)
                run_config = RunConfig(**config.pop('config'), comm=comm)
    else:
        common_comm = comm

    # Get the pipeline configurations
    sub_pipelines = args.pipeline
    pipeline_config = []
    if sub_pipelines is None:
#        sub_pipelines = list(config.keys())
        for sub_pipeline in config.values():
            pipeline_config += sub_pipeline
    else:
        for sub_pipeline in sub_pipelines:
            if sub_pipeline in config:
                pipeline_config += config.get(sub_pipeline)
            else:
                raise ValueError(
                    f'Invalid pipeline option: \'{sub_pipeline}\' missing in '
                    f'the pipeline configuration ({list(config.keys())})')

    # Run the pipeline with or without profiling
    if run_config.profile:
        # System modules
        from cProfile import runctx  # python profiler
        from pstats import Stats     # profiler statistics

        cmd = 'runner(run_config, pipeline_config, common_comm)'
        runctx(cmd, globals(), locals(), 'profile.dat')
        info = Stats('profile.dat')
        info.sort_stats('cumulative')
        info.print_stats()
    else:
        runner(run_config, pipeline_config, common_comm)

    # Disconnect the spawned worker
    if have_mpi and run_config.spawn:
        common_comm.barrier()
        sub_comm.Disconnect()

def runner(run_config, pipeline_config, comm=None):
    """Main runner funtion.

    :param run_config: CHAP run configuration.
    :type run_config: CHAP.runner.RunConfig
    :param pipeline_config: CHAP Pipeline configuration.
    :type pipeline_config: dict
    :param comm: MPI communicator.
    :type comm: mpi4py.MPI.Comm, optional
    :return: The pipeline's returned data field.
    """
    # System modules
    from time import time

    # Logging setup
    logger, log_handler = set_logger(run_config.log_level)
    logger.info(f'Input pipeline configuration: {pipeline_config}\n')

    # Run the pipeline
    t0 = time()
    data = run(run_config, pipeline_config, logger, log_handler, comm)
    logger.info(f'Executed "run" in {time()-t0:.3f} seconds')

    return data

def set_logger(log_level='INFO'):
    """Helper function to set CHAP logger.

    :param log_level: Logger level, defaults to `"INFO"`.
    :type log_level: str
    :return: The CHAP logger and logging handler.
    :rtype: logging.Logger, logging.StreamHandler
    """
    logger = logging.getLogger(__name__)
    log_level = getattr(logging, log_level.upper())
    logger.setLevel(log_level)
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter(
        '{asctime}: {name:20}: {levelname}: {message}',
        datefmt='%Y-%m-%d %H:%M:%S', style='{'))
    logger.addHandler(log_handler)
    return logger, log_handler

def run(
        run_config, pipeline_config, logger=None, log_handler=None, comm=None):
    """Run a given pipeline_config.

    :param run_config: CHAP run configuration.
    :type run_config: CHAP.runner.RunConfig
    :param pipeline_config: CHAP Pipeline configuration.
    :type pipeline_config: dict
    :param logger: CHAP logger.
    :type logger: logging.Logger, optional
    :param log_handler: Logging handler.
    :type log_handler: logging.StreamHandler, optional
    :param comm: MPI communicator.
    :type comm: mpi4py.MPI.Comm, optional
    :return: The `data` field of the first item in the returned
        list of pipeline items.
    """
    # System modules
    from tempfile import NamedTemporaryFile

    # Make sure os.makedirs is only called from the root node
    if comm is None:
        rank = 0
    else:
        rank = comm.Get_rank()

    pipeline_items = []
    pipeline_kwargs = []
    for item in pipeline_config:
        # Load individual object with given name from its module
        kwargs = run_config.model_dump(exclude={'root', 'profile', 'spawn'})
        if isinstance(item, dict):
            name = list(item.keys())[0]
            item_args = item.get(name)
            # Picking "inputdir" and "outputdir" from the item or from
            # the default run configuration, giving precedence to the
            # former
            if 'inputdir' in item_args:
                inputdir = item_args.pop('inputdir')
                if not os.path.isabs(inputdir):
                    inputdir = os.path.normpath(os.path.realpath(
                        os.path.join(run_config.inputdir, inputdir)))
                if not os.path.isdir(inputdir):
                    raise OSError(
                        f'input directory does not exist ({inputdir})')
                if not os.access(inputdir, os.R_OK):
                    raise OSError('input directory is not accessible for '
                                  f'reading ({inputdir})')
                item_args['inputdir'] = inputdir
            if 'outputdir' in item_args:
                outputdir = item_args.pop('outputdir')
                if not os.path.isabs(outputdir):
                    outputdir = os.path.normpath(os.path.realpath(
                        os.path.join(run_config.outputdir, outputdir)))
                if not rank:
                    if not os.path.isdir(outputdir):
                        os.makedirs(outputdir)
                    try:
                        NamedTemporaryFile(dir=outputdir)
                    except Exceptions as exc:
                        raise OSError(
                            'output directory is not accessible for '
                            f'writing ({outputdir})') from exc
                item_args['outputdir'] = outputdir
            kwargs.update(item_args)
        else:
            name = item
        if 'users' in name:
            # Load users module. This is required in CHAPaaS which can
            # have common area for users module. Otherwise, we will be
            # required to have invidual user's PYTHONPATHs to load user
            # processors.
            try:
                # Third party modules
                import users
            except ImportError:
                if logger is not None:
                    logger.error(f'Unable to load {name}')
                continue
            cls_name = name.split('.')[-1]
            mod_name = '.'.join(name.split('.')[:-1])
            module = __import__(mod_name, fromlist=[cls_name])
        else:
            mod_name, cls_name = name.split('.')
            module = __import__(f'CHAP.{mod_name}', fromlist=[cls_name])
        # Initialize the object
        obj = getattr(module, cls_name)(
            inputdir=kwargs.pop('inputdir'),
            outputdir=kwargs.pop('outputdir'),
            interactive=kwargs.pop('interactive'),
            schema=kwargs.pop('schema') if 'schema' in kwargs else None)
        obj.logger.setLevel(kwargs.pop('log_level'))
        if log_handler is not None:
            obj.logger.addHandler(log_handler)
        if logger is not None:
            logger.info(f'Loaded {obj}')
        pipeline_items.append(obj)
        kwargs['comm'] = comm
        pipeline_kwargs.append(kwargs)
    pipeline = Pipeline(pipeline_items, pipeline_kwargs)
    pipeline.logger.setLevel(run_config.log_level)
    if log_handler is not None:
        pipeline.logger.addHandler(log_handler)
    if logger is not None:
        logger.info(f'Loaded {pipeline} with {len(pipeline_items)} items\n')
        logger.info(f'Calling "execute" on {pipeline}')

    # Make sure os.makedirs completes before continuing all nodes
    if comm is not None:
        comm.barrier()

    # Validate the pipeline configuration
    pipeline.validate()

    # Execute the pipeline
    result = pipeline.execute()
    if result:
        return result[0]['data']
    return result


if __name__ == '__main__':
    main()
