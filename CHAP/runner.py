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
            'profile': False}

    def __init__(self, config={}):
        """RunConfig constructor

        :param config: Pipeline configuration options
        :type config: dict
        """
        # System modules
        from tempfile import NamedTemporaryFile

        for opt in self.opts:
            setattr(self, opt, config.get(opt, self.opts[opt]))

        # Check if root exists (create it if not) and is readable
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
        if not os.path.isdir(self.outputdir):
            os.makedirs(self.outputdir)
        try:
            tmpfile = NamedTemporaryFile(dir=self.outputdir)
        except:
            raise OSError('output directory is not accessible for writing '
                          f'({self.outputdir})')

        self.log_level = self.log_level.upper()

def parser():
    """Return an argument parser for the `CHAP` CLI. This parser has
    one argument: the input CHAP configuration file.
    """
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument(
        'config', action='store', default='',
        help='Input configuration file')
    return parser

def main():
    """Main function"""
    args = parser().parse_args()

    # read input config file
    configfile = args.config
    with open(configfile) as file:
        config = safe_load(file)
    run_config = RunConfig(config.get('config', {}))
    pipeline_config = config.get('pipeline', [])

    # profiling setup
    if run_config.profile:
        from cProfile import runctx  # python profiler
        from pstats import Stats     # profiler statistics
        cmd = 'runner(run_config, pipeline_config)'
        runctx(cmd, globals(), locals(), 'profile.dat')
        info = Stats('profile.dat')
        info.sort_stats('cumulative')
        info.print_stats()
    else:
        runner(run_config, pipeline_config)

def runner(run_config, pipeline_config):
    """Main runner funtion

    :param run_config: CHAP run configuration
    :type run_config: RunConfig
    :param pipeline_config: CHAP Pipeline configuration
    :type pipeline_config: dict
    """
    # logging setup
    logger, log_handler = setLogger(run_config.log_level)
    logger.info(f'Input pipeline configuration: {pipeline_config}\n')

    # run pipeline
    run(pipeline_config,
        run_config.inputdir, run_config.outputdir, run_config.interactive,
        logger, run_config.log_level, log_handler)

def setLogger(log_level="INFO"):
    """
    Helper function to set CHAP logger

    :param log_level: logger level, default INFO
    """
    logger = logging.getLogger(__name__)
    log_level = getattr(logging, log_level.upper())
    logger.setLevel(log_level)
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter(
        '{name:20}: {message}', style='{'))
    logger.addHandler(log_handler)
    return logger, log_handler

def run(
        pipeline_config, inputdir=None, outputdir=None, interactive=False,
        logger=None, log_level=None, log_handler=None):
    """
    Run given pipeline_config

    :param pipeline_config: CHAP pipeline config
    """
    # System modules
    from tempfile import NamedTemporaryFile

    objects = []
    kwds = []
    for item in pipeline_config:
        # load individual object with given name from its module
        kwargs = {'inputdir': inputdir,
                  'outputdir': outputdir,
                  'interactive': interactive}
        if isinstance(item, dict):
            name = list(item.keys())[0]
            item_args = item[name]
            # Combine the function's input arguments "inputdir",
            # "outputdir" and "interactive" with the item's arguments
            # joining "inputdir" and "outputdir" and giving precedence
            # for "interactive" in the latter
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
                if not os.path.isdir(newoutputdir):
                    os.makedirs(newoutputdir)
                try:
                    tmpfile = NamedTemporaryFile(dir=newoutputdir)
                except:
                    raise OSError('output directory is not accessible for '
                                  f'writing ({newoutputdir})')
                kwargs['outputdir'] = newoutputdir
            kwargs = {**kwargs, **item_args}
        else:
            name = item
        if "users" in name:
            # load users module. This is required in CHAPaaS which can
            # have common area for users module. Otherwise, we will be
            # required to have invidual user's PYTHONPATHs to load user
            # processors.
            try:
                import users
            except ImportError:
                if logger:
                    logger.error(f'Unable to load {name}')
                continue
            clsName = name.split('.')[-1]
            modName = '.'.join(name.split('.')[:-1])
            module = __import__(modName, fromlist=[clsName])
            obj = getattr(module, clsName)()
        else:
            modName, clsName = name.split('.')
            module = __import__(f'CHAP.{modName}', fromlist=[clsName])
            obj = getattr(module, clsName)()
        if log_level:
            obj.logger.setLevel(log_level)
        if log_handler:
            obj.logger.addHandler(log_handler)
        if logger:
            logger.info(f'Loaded {obj}')
        objects.append(obj)
        kwds.append(kwargs)
    pipeline = Pipeline(objects, kwds)
    if log_level:
        pipeline.logger.setLevel(log_level)
    if log_handler:
        pipeline.logger.addHandler(log_handler)
    if logger:
        logger.info(f'Loaded {pipeline} with {len(objects)} items\n')
        logger.info(f'Calling "execute" on {pipeline}')
    pipeline.execute()


if __name__ == '__main__':
    main()
