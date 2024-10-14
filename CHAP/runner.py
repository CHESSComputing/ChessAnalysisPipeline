"""
File       : runner.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description:
"""

# system modules
import argparse
import logging
import os
from yaml import safe_load

# local modules
from CHAP.pipeline import Pipeline


class RunConfig():
    """Representation of Pipeline run configuration."""
    opts = {'root': os.getcwd(),
            'inputdir': '.',
            'outputdir': '.',
            'interactive': False,
            'log_level': 'INFO',
            'profile': False,
            'spawn': 0}

    def __init__(self, config=None, comm=None):
        """RunConfig constructor.

        :param config: Pipeline configuration options.
        :type config: dict, optional
        :param comm: MPI communicator.
        :type comm: mpi4py.MPI.Comm, optional
        """
        # System modules
        from tempfile import NamedTemporaryFile

        # Make sure os.makedirs is only called from the root node
        if comm is None:
            rank = 0
        else:
            rank = comm.Get_rank()
        if config is None:
            config = {}
        for opt in self.opts:
            setattr(self, opt, config.get(opt, self.opts[opt]))

        # Check if root exists (create it if not) and is readable
        if not rank:
            if not os.path.isdir(self.root):
                os.makedirs(self.root)
            if not os.access(self.root, os.R_OK):
                raise OSError('root directory is not accessible for reading '
                              f'({self.root})')

        # Check if inputdir exists and is readable
        if not os.path.isabs(self.inputdir):
            self.inputdir = os.path.realpath(
                os.path.join(self.root, self.inputdir))
        if not os.path.isdir(self.inputdir):
            raise OSError(f'input directory does not exist ({self.inputdir})')
        if not os.access(self.inputdir, os.R_OK):
            raise OSError('input directory is not accessible for reading '
                          f'({self.inputdir})')

        # Check if outputdir exists (create it if not) and is writable
        if not os.path.isabs(self.outputdir):
            self.outputdir = os.path.realpath(
                os.path.join(self.root, self.outputdir))
        if not rank:
            if not os.path.isdir(self.outputdir):
                os.makedirs(self.outputdir)
            try:
                NamedTemporaryFile(dir=self.outputdir)
            except:
                raise OSError('output directory is not accessible for writing '
                              f'({self.outputdir})')

        self.log_level = self.log_level.upper()

        # Make sure os.makedirs completes before continuing all nodes
        if comm is not None:
            comm.barrier()

def parser():
    """Return an argument parser for the `CHAP` CLI. This parser has
    one argument: the input CHAP configuration file.
    """
    pparser = argparse.ArgumentParser(prog='PROG')
    pparser.add_argument(
        'config', action='store', default='', help='Input configuration file')
    pparser.add_argument(
        '-p', '--pipeline', nargs='*', help='Pipeline name(s)')
    return pparser

def main():
    """Main function."""
    # Third party modules
    try:
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

    # Check if run was a worker spawned by another Processor
    run_config = RunConfig(config.pop('config'), comm)
    if have_mpi and run_config.spawn:
        sub_comm = MPI.Comm.Get_parent()
        common_comm = sub_comm.Merge(True)
        # Read worker specific input config file
        if run_config.spawn > 0:
            with open(f'{configfile}_{common_comm.Get_rank()}') as file:
                config = safe_load(file)
                run_config = RunConfig(config.get('config'), common_comm)
        else:
            with open(f'{configfile}_{sub_comm.Get_rank()}') as file:
                config = safe_load(file)
                run_config = RunConfig(config.get('config'), comm)
    else:
        common_comm = comm

    # Get the pipeline configurations
    pipeline = args.pipeline
    pipeline_config = []
    if pipeline is None:
        for sub_pipeline in config.values():
            pipeline_config += sub_pipeline
    else:
        for sub_pipeline in pipeline:
            if sub_pipeline in config:
                pipeline_config += config.get(sub_pipeline)
            else:
                raise ValueError(
                    f'Invalid pipeline option: \'{sub_pipeline}\' missing in '
                    f'the pipeline configuration ({list(config.keys())})')

    # Profiling setup
    if run_config.profile:
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

    # logging setup
    logger, log_handler = set_logger(run_config.log_level)
    logger.info(f'Input pipeline configuration: {pipeline_config}\n')

    # Run the pipeline
    t0 = time()
    data = run(pipeline_config,
        run_config.inputdir, run_config.outputdir, run_config.interactive,
        logger, run_config.log_level, log_handler, comm)
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
        pipeline_config, inputdir=None, outputdir=None, interactive=False,
        logger=None, log_level=None, log_handler=None, comm=None):
    """Run a given pipeline_config.

    :param pipeline_config: CHAP Pipeline configuration.
    :type pipeline_config: dict
    :param inputdir: Input directory.
    :type inputdir: str, optional
    :param outputdir: Output directory.
    :type outputdir: str, optional
    :param interactive: Allows for user interactions,
        defaults to `False`.
    :type interactive: bool, optional
    :param logger: CHAP logger.
    :type logger: logging.Logger, optional
    :param log_level: Logger level.
    :type log_level: str, optional
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

    objects = []
    kwds = []
    for item in pipeline_config:
        # Load individual object with given name from its module
        kwargs = {'inputdir': inputdir,
                  'outputdir': outputdir,
                  'interactive': interactive,
                  'comm': comm}
        if isinstance(item, dict):
            name = list(item.keys())[0]
            item_args = item[name]
            # Combine the function's input arguments "inputdir",
            # "outputdir" and "interactive" with the item's arguments
            # joining "inputdir" and "outputdir" and giving precedence
            # for "interactive" in the latter
            if item_args is not None:
                if 'inputdir' in item_args:
                    newinputdir = os.path.normpath(os.path.join(
                        kwargs['inputdir'], item_args.pop('inputdir')))
                    if not os.path.isdir(newinputdir):
                        raise OSError(
                            f'input directory does not exist ({newinputdir})')
                    if not os.access(newinputdir, os.R_OK):
                        raise OSError('input directory is not accessible for '
                                      f'reading ({newinputdir})')
                    kwargs['inputdir'] = newinputdir
                if 'outputdir' in item_args:
                    newoutputdir = os.path.normpath(os.path.join(
                        kwargs['outputdir'], item_args.pop('outputdir')))
                    if not rank:
                        if not os.path.isdir(newoutputdir):
                            os.makedirs(newoutputdir)
                        try:
                            NamedTemporaryFile(dir=newoutputdir)
                        except Exceptions as exc:
                            raise OSError(
                                'output directory is not accessible for '
                                f'writing ({newoutputdir})') from exc
                    kwargs['outputdir'] = newoutputdir
                kwargs = {**kwargs, **item_args}
        else:
            name = item
        if 'users' in name:
            # Load users module. This is required in CHAPaaS which can
            # have common area for users module. Otherwise, we will be
            # required to have invidual user's PYTHONPATHs to load user
            # processors.
            try:
                import users
            except ImportError:
                if logger is not None:
                    logger.error(f'Unable to load {name}')
                continue
            cls_name = name.split('.')[-1]
            mod_name = '.'.join(name.split('.')[:-1])
            module = __import__(mod_name, fromlist=[cls_name])
            obj = getattr(module, cls_name)()
        else:
            mod_name, cls_name = name.split('.')
            module = __import__(f'CHAP.{mod_name}', fromlist=[cls_name])
            obj = getattr(module, cls_name)()
        if log_level is not None:
            obj.logger.setLevel(log_level)
        if log_handler is not None:
            obj.logger.addHandler(log_handler)
        if logger is not None:
            logger.info(f'Loaded {obj}')
        objects.append(obj)
        kwds.append(kwargs)
    pipeline = Pipeline(objects, kwds)
    if log_level is not None:
        pipeline.logger.setLevel(log_level)
    if log_handler is not None:
        pipeline.logger.addHandler(log_handler)
    if logger is not None:
        logger.info(f'Loaded {pipeline} with {len(objects)} items\n')
        logger.info(f'Calling "execute" on {pipeline}')

    # Make sure os.makedirs completes before continuing all nodes
    if comm is not None:
        comm.barrier()

    return pipeline.execute()[0]['data']


if __name__ == '__main__':
    main()
