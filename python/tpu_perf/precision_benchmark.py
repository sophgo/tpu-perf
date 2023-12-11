import os
import sys
import csv
import logging
from .buildtree import check_buildtree, BuildTree
from .logger import init_logger
from .harness import load_plugins
load_plugins()

class CSVWrapper:
    def __init__(self, fd):
        self.fd = fd
        self.writer = csv.writer(fd)

    def writerow(self, *args, **kw_args):
        self.writer.writerow(*args, **kw_args)
        self.fd.flush()

import ctypes
def malloc_trim():
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError as err:
        logging.error(f'{err}')

class Runner:
    def __init__(self):
        self.stat_files = dict()
        self.tested_names = set()

    def run(self, tree, path, config):
        if 'harness' not in config:
            return
        from .harness import get_harness
        key = config['harness']['type']
        harness = get_harness(key)

        def get_csv(stats):
            if key not in self.stat_files:
                fn = os.path.join(tree.global_config['outdir'], f'{key}.csv')
                self.stat_files[key] = CSVWrapper(open(fn, 'w'))
                csv_f = self.stat_files[key]
                csv_f.writerow(['name'] + list(stats.keys()))
            else:
                csv_f = self.stat_files[key]
            return csv_f

        for args in config['harness']['args']:
            for num_core in config['core_list']:
                config['num_core'] = num_core
                bmodel = tree.expand_variables(config, args['bmodel'])
                if not os.path.exists(bmodel):
                    logging.warning(f'{bmodel} does not exist')
                    continue
                name = tree.expand_variables(config, args['name'])
                name_suffix = '' if num_core == 1 else f'-{num_core}core'
                name = f'{config["name"]}-{name}{name_suffix}'
                if name in self.tested_names:
                    logging.warning(f'Skip duplicate {name}')
                    continue
                self.tested_names.add(name)
                stats = harness(tree, config, args)
                malloc_trim()
                get_csv(stats).writerow([name] + [
                    f'{v:.2%}' if type(v) == float else str(v)
                    for v in stats.values()])

def main():
    init_logger()

    if not check_buildtree():
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description='tpu-perf benchmark tool')
    BuildTree.add_arguments(parser)
    args = parser.parse_args()

    tree = BuildTree(os.path.abspath('.'), args)
    runner = Runner()
    for path, config in tree.walk():
        runner.run(tree, path, config)

if __name__ == '__main__':
    main()
