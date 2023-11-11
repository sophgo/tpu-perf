import sys
import os
import logging
import shutil
import re
from .buildtree import check_buildtree, BuildTree
from .subp import CommandExecutor, sys_memory_size
from .util import *
import time
import json

def replace_shape_batch(cmd, batch_size):
    match = re.search('-shapes(=| *)["\']?((\[[\\d, ]+\],?)+)["\']?', cmd)
    if match is None:
        logging.error(f'failed to find --shapes [n,c,h,w] in "{cmd}"')
        raise RuntimeError('Invalid argument')
    shapes_str = match.group(2)
    shapes = eval(shapes_str)
    if type(shapes[0]) != list:
        shapes = [shapes]
    for s in shapes:
        s[0] = batch_size
    new_str = ','.join(
        f'[{",".join(str(i) for i in s)}]'
        for s in shapes)
    return cmd.replace(shapes_str, new_str)

option_time_only = False

import glob
from functools import reduce

def files_equal(inputs, output):
    if not os.path.exists(output):
        return False
    input_size = reduce(lambda acc, x: acc + os.stat(x).st_size, inputs, 0)
    output_size = os.stat(output).st_size
    return input_size == output_size

def build_common(tree, path, config):
    concat_files = config.get('concat_files', [])
    workdir = config['workdir']
    pool = CommandExecutor(workdir)
    for i, concat in enumerate(concat_files):
        inputs = concat.get('inputs')
        if not inputs or type(inputs) != list:
            logging.error(f'Invalid inputs in concat_files[{i}]')
            raise RuntimeError('Invalid argument')
        output = concat.get('output')
        if not output or type(output) != str:
            logging.error(f'Invalid output in concat_files[{i}]')
            raise RuntimeError('Invalid argument')
        inputs = [tree.expand_variables(config, i) for i in inputs]
        inputs = reduce(lambda acc, x: acc + glob.glob(x), inputs, [])
        if not inputs:
            logging.error(f'Invalid inputs in concat_files[{i}], files not found')
            raise RuntimeError('Invalid argument')
        output = tree.expand_variables(config, output)
        if not output.startswith('/'):
            output = os.path.join(workdir, output)
        if files_equal(inputs, output):
            logging.info(f'{output} already concated')
        else:
            inputs.sort()
            cmd = f'cat {" ".join(inputs)} > {output}'
            pool.put(f'file_concat-{i}', cmd)
    pool.wait()

def build_mlir(tree, path, config):
    build_common(tree, path, config)

    workdir = config['workdir']
    name = config['name']
    if config['model_name'] and name != config['model_name']:
        return
    env = [
        tree.expand_variables(config, v)
        for v in config.get('mlir_build_env', [])]
    pool = CommandExecutor(workdir, env)

    if 'mlir_transform' in config:
        logging.info(f'Transforming MLIR {name}...')
        trans_cmd = tree.expand_variables(config, config['mlir_transform'])
        pool.put('mlir_transform', trans_cmd)
        pool.wait()
        logging.info(f'Transform MLIR {name} done')

    if 'mlir_calibration' in config:
        logging.info(f'Calibrating MLIR {name}...')
        cali_cmd = tree.expand_variables(config, config['mlir_calibration'])
        pool.put('mlir_calibration', cali_cmd)
        pool.wait()
        logging.info(f'Calibrate MLIR {name} done')

    for i in range(len(config['core_list'])):
        config['num_core'] = config['core_list'][i]

        if 'deploy' in config:
            logging.info(f'Deploying {name}_{config["num_core"]}_core...')
            if type(config['deploy']) != list:
                config['deploy'] = [config['deploy']]
            fns = [fn for fn in os.listdir(workdir) if fn.endswith('npz')]
            for j, deploy in enumerate(config['deploy']):
                title = f'mlir_deploy_core{config["num_core"]}.{j}'
                cwd = os.path.join(workdir, title)
                os.makedirs(cwd, exist_ok=True)
                for fn in fns:
                    shutil.copyfile(
                        os.path.join(workdir, fn),
                        os.path.join(cwd, fn))
                pool.put(
                    title,
                    tree.expand_variables(config, deploy),
                    cwd=cwd)
                pool.wait()
            logging.info(f'Deploy {name}_{config["num_core"]}_core done')

def build_nntc(tree, path, config):
    build_common(tree, path, config)

    workdir = config['workdir']
    if option_time_only and not config.get('time', True):
        return
    if not option_time_only and not config.get('precision'):
        return
    env = [
        tree.expand_variables(config, v)
        for v in config.get('build_env', [])]
    name = config['name']
    if 'shape_key' in config:
        name = f'{name}-{config["shape_key"]}'

    int8_pool = CommandExecutor(
        workdir, env,
        memory_hint=config.get('memory_hint'))

    cali_key = 'time_only_cali' if option_time_only else 'cali'

    if cali_key in config:
        # Calibrate

        # Auto cali uses model path, hack it
        # TODO
        model_path = tree.expand_variables(config, config['model'])
        model_fn = os.path.basename(model_path)
        hack_model_path = os.path.join(workdir, model_fn)
        if os.path.isdir(model_path):
            if os.path.isdir(hack_model_path):
                shutil.rmtree(hack_model_path)
            shutil.copytree(model_path, hack_model_path)
        else:
            shutil.copyfile(model_path, hack_model_path)
        config['model'] = hack_model_path

        cmd = tree.expand_variables(config, config[cali_key])
        logging.info(f'Calibrating {name}...')
        start = time.monotonic()
        int8_pool.run(cali_key, cmd)
        elaps = format_seconds(time.monotonic() - start)
        logging.info(f'{name} calibration finished in {elaps}.')

    if 'fp_compile_options' in config:
        fp_pool = CommandExecutor(workdir, env)
        start = time.monotonic()
        logging.info(f'Building float bmodel {name}...')
        batch_sizes = config.get('fp_batch_sizes', [1]) \
            if option_time_only else [1]
        fp_loops = config.get('fp_loops') or \
            tree.global_config.get('fp_loops') or [dict()]
        for loop in fp_loops:
            loop_config = dict_override(config, loop)
            cmd = tree.expand_variables(loop_config, loop_config["fp_compile_options"])
            for batch_size in batch_sizes:
                if 'fp_batch_sizes' in config:
                    batch_cmd = replace_shape_batch(cmd, batch_size)
                else:
                    batch_cmd = cmd
                outdir = loop_config.get(
                    'fp_outdir_template',
                    '{}b.fp.compilation').format(batch_size)
                env = [
                    tree.expand_variables(loop_config, v)
                    for v in loop_config.get('build_env', [])]
                fp_pool.put(
                    outdir,
                    f'{batch_cmd} --outdir {outdir}',
                    env=env)
        fp_pool.wait()
        elaps = format_seconds(time.monotonic() - start)
        logging.info(f'float bmodel {name} done in {elaps}.')

    if 'bmnetu_options' in config:
        # Build int8 bmodel
        start = time.monotonic()
        logging.info(f'Compiling {name}...')
        int8_loops = config.get('int8_loops') or \
            tree.global_config.get('int8_loops') or [dict()]
        for loop in int8_loops:
            loop_config = dict_override(config, loop)
            cmd = f'python3 -m bmnetu {loop_config["bmnetu_options"]}'
            cmd = tree.expand_variables(loop_config, cmd)
            for b in loop_config['bmnetu_batch_sizes']:
                outdir = loop_config.get(
                    'int8_outdir_template', '{}b.compilation').format(b)
                env = [
                    tree.expand_variables(loop_config, v)
                    for v in loop_config.get('build_env', [])]
                int8_pool.put(
                    outdir,
                    f'{cmd} --max_n {b} --outdir {outdir}',
                    env=env)
        int8_pool.wait()
        elaps = format_seconds(time.monotonic() - start)
        logging.info(f'INT8 bmodel {name} done in {elaps}.')

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s %(filename)s:%(lineno)d] %(message)s')

    if not check_buildtree():
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description='tpu-perf benchmark tool')
    parser.add_argument('--time', action='store_true')
    parser.add_argument('--exit-on-error', action='store_true')
    parser.add_argument('--report', type=str, help='report model compilation results to the specified json file')
    BuildTree.add_arguments(parser)
    args = parser.parse_args()
    if args.num_core != 1:
        assert args.target in ['BM1688', 'CV186X']
    global option_time_only
    option_time_only = args.time

    tree = BuildTree(os.path.abspath('.'), args)

    mem_size = sys_memory_size()
    max_workers = max(1, int(mem_size / 1024 / 1024 / 12))
    num_workers = 4
    if num_workers > max_workers:
        num_workers = max_workers
        logging.info(
            f'System memory {mem_size}, using {num_workers} workers.')

    build_fn = build_mlir if args.mlir else build_nntc

    from concurrent.futures import ThreadPoolExecutor, as_completed
    ret = 0
    futures = {}
    succ_cases, failed_cases = [], []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for path, config in tree.walk():
            future = executor.submit(build_fn, tree, path, config)
            if config['model_name'] and config['name'] != config['model_name']:
                continue
            futures[config['name']] = future

        for f in as_completed(futures.values()):
            try:
                result = f.result()
                succ_cases.append(next(key for key, value in futures.items() if value == f))
            except Exception as err:
                if args.exit_on_error:
                    logging.error(f'Quit because of exception, {err}')
                    os._exit(-1)
                else:
                    logging.warning(f'Task failed, {err}')
                    failed_cases.append(next(key for key, value in futures.items() if value == f))
                    ret = -1

        if args.report:
            output_fn = f'{args.report}'
            params = {"succ_cases": list(set(succ_cases)), "failed_cases": list(set(failed_cases))}
            with open(output_fn, 'w') as f:
                json.dump(params, f)

    sys.exit(ret)

if __name__ == '__main__':
    main()
