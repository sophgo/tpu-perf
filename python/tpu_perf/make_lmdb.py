import sys
import os
import logging
from .buildtree import check_buildtree, BuildTree
from .subp import sys_memory_size

from threading import Lock

lock = Lock()

from .preprocess import load_plugins
load_plugins()

def build_lmdb(data_dir, tree, path, config):
    from .preprocess import get_preprocess_method
    if 'input' not in config:
        return
    data_config = config['input']
    name = data_config['dataset']
    if name not in tree.global_config:
        logging.error(f'Invalid dataset {name}')
        raise RuntimeError('invalid argument')
    data_path = tree.read_global_variable(name)
    preprocess = get_preprocess_method(data_config['preprocess'])
    out_path = config['lmdb_out']

    with lock:
        if os.path.exists(os.path.join(out_path, 'info.yaml')):
            logging.info(f'{config["name"]} {out_path} already exist')
            return
        os.makedirs(out_path, exist_ok=True)
        info_fn = os.path.join(out_path, 'info.yaml')
        import yaml
        with open(info_fn, 'w') as f:
            yaml.dump(data_config, f)

    old_cwd = os.getcwd()
    os.chdir(out_path)
    try:
        preprocess(data_path, out_path, tree, config)
    finally:
        os.chdir(old_cwd)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s %(filename)s:%(lineno)d] %(message)s')

    if not check_buildtree():
        sys.exit(1)
    import argparse
    parser = argparse.ArgumentParser(description='tpu-perf benchmark tool')
    BuildTree.add_arguments(parser)
    args = parser.parse_args()
    tree = BuildTree(os.path.abspath('.'), args)

    # Prepare data path
    try:
        data_dir = tree.read_global_variable('data_dir')
    except KeyError:
        logging.error(f'Please specify data_dir in root config.yaml')
        sys.exit(1)
    os.makedirs(data_dir, exist_ok=True)

    mem_size = sys_memory_size()
    max_workers = max(1, int(mem_size / 1024 / 1024 / 7))
    num_workers = 4
    if num_workers > max_workers:
        num_workers = max_workers
        logging.info(
            f'System memory {mem_size}, using {num_workers} workers.')

    # Synchronous invoking for debug
    #
    #for path, config in tree.walk():
    #    build_lmdb(data_dir, tree, path, config)
    #return

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for path, config in tree.walk():
            f = executor.submit(build_lmdb, data_dir, tree, path, config)
            futures.append(f)

        for f in as_completed(futures):
            err = f.exception()
            if err:
                logging.error(f'Quit because of exception, {err}')
                os._exit(-1)

if __name__ == '__main__':
    main()
