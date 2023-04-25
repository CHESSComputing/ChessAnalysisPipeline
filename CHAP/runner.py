"""
File       : runner.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description:
"""

# system modules
import argparse
import logging
from yaml import safe_load

# local modules
from CHAP.pipeline import Pipeline


class OptionParser():
    """User based option parser"""
    def __init__(self):
        """OptionParser class constructor"""
        self.parser = argparse.ArgumentParser(prog='PROG')
        self.parser.add_argument(
            '--config', action='store', dest='config', default='',
            help='Input configuration file')
        self.parser.add_argument(
            '--interactive', action='store_true', dest='interactive',
            help='Allow interactive processes')
        self.parser.add_argument(
            '--log-level', choices=logging._nameToLevel.keys(),
            dest='log_level', default='INFO', help='logging level')
        self.parser.add_argument(
            '--profile', action='store_true', dest='profile',
            help='profile output')


def main():
    """Main function"""
    optmgr = OptionParser()
    opts = optmgr.parser.parse_args()
    if opts.profile:
        from cProfile import runctx  # python profiler
        from pstats import Stats     # profiler statistics
        cmd = 'runner(opts)'
        runctx(cmd, globals(), locals(), 'profile.dat')
        info = Stats('profile.dat')
#        info.strip_dirs()
        info.sort_stats('cumulative')
        info.print_stats()
    else:
        runner(opts)


def runner(opts):
    """Main runner function

    :param opts: object containing input parameters
    :type opts: OptionParser
    """

    log_level = opts.log_level.upper()
    logger, log_handler = setLogger(log_level)
    config = {}
    with open(opts.config) as file:
        config = safe_load(file)
    logger.info(f'Input configuration: {config}\n')
    pipeline_config = config.get('pipeline', [])
    run(pipeline_config, opts.interactive, logger, log_level, log_handler)

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

def run(pipeline_config, interactive=False, logger=None, log_level=None, log_handler=None):
    """
    Run given pipeline_config

    :param pipeline_config: CHAP pipeline config
    """
    objects = []
    kwds = []
    for item in pipeline_config:
        # load individual object with given name from its module
        kwargs = {'interactive': interactive}
        if isinstance(item, dict):
            name = list(item.keys())[0]
            # Combine the "interactive" command line argument with the object's keywords
            # giving precedence of "interactive" in the latter
            kwargs = {**kwargs, **item[name]}
        else:
            name = item
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
