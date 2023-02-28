import os
import re
import csv
import math
import psutil
import sys
import time
import logging
from .buildtree import check_buildtree, BuildTree
from .subp import CommandExecutor
from .util import *

option_cmodel_stats = False

class Average:
    def __init__(self):
        self.clear()

    def put(self, v):
        self.acc += v
        self.count += 1

    def get(self):
        return self.acc / self.count

    def clear(self):
        self.acc = 0
        self.count = 0

def parse_stats(string):
    time_prog = 'INFO:(.+) time\(s\): ([\.\d]+)'
    ret = dict()
    for k, v in re.findall(time_prog, string):
        k = k.strip().replace(' ', '_')
        if k not in ret:
            ret[k] = Average()
        ret[k].put(float(v))
    for k in ret.keys():
        ret[k] = ret[k].get()

    shape_prog = 'Input \d+\).+shape=\[([\d ]+)\]'
    shape_info = ':'.join(
        'x'.join(s.split()) for s in re.findall(shape_prog, string))
    ret['shape'] = shape_info

    return ret

def parse_profile(fn):
    with open(fn) as f:
        lines = f.read()
    if not lines:
        return
    lines = lines[lines.find('API_END'):]
    data = dict()
    for pair in re.finditer('(\w+) *: *([\d\.]+)', lines):
        v = pair.group(2)
        data[pair.group(1)] = float(v) if '.' in v else int(v)
    return data

def format_float(v):
    if v > 0.1:
        return f'{v:.03f}'
    else:
        return f'{v:.03e}'

def run_model(tree, config, name, b, profile_path, bmodel, stat_f, extra):
    title = f'run.{name}'
    workdir = config['workdir']
    env = [
        tree.expand_variables(config, v)
        for v in config.get('run_env', [])]
    env.append('BMRUNTIME_PROFILE_OUT_DIR={b}b.profiledata')
    pool = CommandExecutor(workdir, env)
    rt_cmp = config.get('runtime_cmp')
    iter_opt = tree.global_config.get('iter_opt', '--loopnum')
    if 'iter_opt' in config:
        iter_opt = config['iter_opt']
    bmodel_dir = os.path.dirname(bmodel)

    info = None
    rounds = None
    if os.path.exists(profile_path):
        info = parse_profile(profile_path)
        if info is not None:
            rounds = int(1200 / info['runtime'])
            max_rounds = 10000
            min_rounds = 1
            if rounds > max_rounds:
                rounds = max_rounds
            if rounds < min_rounds:
                rounds = min_rounds
    if 'time_rounds' in config:
        rounds = math.ceil(config['time_rounds'] / b)
    elif rounds is None:
        rounds = 2000 / b

    full_name = f'{config["name"]} {name}'
    logging.info(f'Run {rounds} times for {full_name}')

    ref_fn = os.path.join(bmodel_dir, 'output_ref_data.dat')
    dev = tree.global_config['devices'][0]
    cmd_opts = ['bmrt_test', iter_opt, str(rounds), '--dev', str(dev)]
    if rt_cmp and os.path.exists(ref_fn) and os.path.getsize(ref_fn):
        logging.info(f'Runtime test {full_name}')
        pool.put(
            title,
            [*cmd_opts, '--context', bmodel_dir],
            shell=False)
    else:
        logging.info(f'Runtime test {full_name} without reference')
        pool.put(
            title,
            [*cmd_opts, '--bmodel', bmodel],
            shell=False)
    try:
        pool.fire()
        pid = pool.pipes[0].pid
        p = psutil.Process(pid)
        cpu_percent = p.cpu_percent(interval=1) / 100
        pool.drain()
        pool.procs.clear()
    except RuntimeError:
        logging.error(f'Runtime test {full_name} failed')

    log_fn = os.path.join(workdir, f'{title}.log')
    with open(log_fn) as f:
        stats = parse_stats(f.read())
    from math import nan
    real_time = stats['calculate'] * 1000 if 'calculate' in stats else nan
    if 'calculate_times' in iter_opt:
        real_time /= rounds
    row = [
        config['name'],
        *[config.get(k, '') for k in extra],
        stats['shape'],
        format_float(config['gops'] * b) if 'gops' in config else 'N/A',
        format_float(real_time)]

    # If profile exists, calculate mac & ddr utilization
    if tree.global_config['target'] == 'BM1684':
        if config['prec'] == 'FP32':
            mac_total = 2.2
        elif config['prec'] == 'INT8':
            mac_total = 17.6
        else:
            logging.error(f'Invalid prec type "{config["prec"]}" for BM1684')
            raise RuntimeError('Invalid prec')
        ddr_total = 32
    elif tree.global_config['target'] == 'BM1684X':
        if config['prec'] == 'FP32':
            mac_total = 2
        elif config['prec'] == 'FP16' or config['prec'] == 'BF16':
            mac_total = 16
        elif config['prec'] == 'INT8':
            mac_total = 32
        else:
            logging.error(f'Invalid prec type "{config["prec"]}" for BM1684')
            raise RuntimeError('Invalid prec')
        ddr_total = 64
    else:
        logging.error(f'Invalid target {tree.global_config["target"]}')
        raise RuntimeError('Invalid target')
    if info is not None:
        s2l = info['S2L']
        l2s = info['L2S']
        s2s = info['S2S']
        calc_mac_util = lambda t: config['gops'] * b / t / mac_total
        calc_ddr_bandwidth = lambda t: \
            (s2l + l2s + s2s * 2) / t * 1000 / 1024**3 / ddr_total

        est_time = info['runtime']
        if option_cmodel_stats:
            row.append(format_float(est_time))
        cpu_index = len(row) + 1
        if 'gops' not in config:
            logging.warning(
                f'Profile exists but no GOPs in config.yaml, {config["name"]}')
            row.append('N/A')
            if option_cmodel_stats:
                row.append('N/A')
        else:
            row.append(f'{calc_mac_util(real_time):.2%}')
            if option_cmodel_stats:
                row.append(f'{calc_mac_util(est_time):.2%}')
        row.insert(cpu_index, f'{cpu_percent:.2%}')
        row.append(f'{calc_ddr_bandwidth(real_time):.2%}')
        if option_cmodel_stats:
            row.append(f'{calc_ddr_bandwidth(est_time):.2%}')
    else:
        ext = ['N/A'] * (5 if option_cmodel_stats else 2)
        cpu_index = 2 if option_cmodel_stats else 1
        ext.insert(cpu_index, f'{cpu_percent:.2%}')
        row.extend(ext)

    stat_f.writerow(row)

def run_mlir(tree, path, raw_config, stat_f, extra):
    workdir = raw_config['workdir']
    for dirpath, dirnames, filenames in os.walk(workdir):
        for fn in filenames:
            if not fn.endswith('.bmodel'):
                continue
            if fn.find('compilation') >= 0:
                continue
            name = os.path.splitext(fn)[0]
            bmodel = os.path.join(dirpath, fn)
            profile_path = bmodel + '.compiler_profile_0.txt'
            config = raw_config.copy()
            config['name'] = name
            if 'prec' not in config:
                config['prec'] = 'INT8' if 'int8' in name else 'FP32'
            run_model(
                tree, config,
                name,
                1,
                profile_path,
                bmodel,
                stat_f, extra)

def run_nntc(tree, path, raw_config, stat_f, extra):
    if not raw_config.get('time', True):
        return
    workdir = raw_config['workdir']

    profile_fn = 'compiler_profile_0.dat' \
        if tree.global_config['target'] == 'BM1684' else \
        'compiler_profile_0.txt'
    fp_loops = raw_config.get('fp_loops') or \
        tree.global_config.get('fp_loops') or [dict()]
    for loop in fp_loops:
        if 'fp_compile_options' not in raw_config:
            # Skip fp bmrt test
            break
        config = dict_override(raw_config, loop)
        batch_sizes = config.get('fp_batch_sizes', [1])
        for b in batch_sizes:
            name = config.get('fp_outdir_template', '{}b.fp.compilation').format(b)
            bmodel_dir = os.path.join(workdir, name)
            bmodel = os.path.join(bmodel_dir, 'compilation.bmodel')
            if not os.path.exists(bmodel):
                logging.warning(f'{bmodel} does not exist')
                continue
            profile_path = os.path.join(bmodel_dir, profile_fn)
            if 'prec' not in config:
                config['prec'] = 'FP32'
            run_model(
                tree, config, name, b, profile_path,
                bmodel, stat_f, extra)

    int8_loops = raw_config.get('int8_loops') or \
        tree.global_config.get('int8_loops') or [dict()]
    for loop in int8_loops:
        if 'bmnetu_options' not in raw_config:
            # Skip bmrt test
            break
        config = dict_override(raw_config, loop)
        for b in config['bmnetu_batch_sizes']:
            name = config.get('int8_outdir_template', '{}b.compilation').format(b)
            bmodel_dir = os.path.join(workdir, name)
            bmodel = os.path.join(bmodel_dir, 'compilation.bmodel')
            if not os.path.exists(bmodel):
                logging.warning(f'{bmodel} does not exist')
                continue
            profile_path = os.path.join(bmodel_dir, profile_fn)
            if 'prec' not in config:
                config['prec'] = 'INT8'
            run_model(
                tree, config, name, b, profile_path,
                bmodel, stat_f, extra)

def collect_nntc_headers(tree, config):
    extra = set(['prec'])
    for loop in config.get('fp_loops', [dict()]):
        for k in loop.keys():
            extra.add(k)
    for loop in config.get('int8_loops', [dict()]):
        for k in loop.keys():
            extra.add(k)
    def skip_if(k):
        if 'template' in k:
            return True
        if k in {'build_env'}:
            return True
    return set(k for k in extra if not skip_if(k))

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s %(filename)s:%(lineno)d] %(message)s')

    import argparse
    parser = argparse.ArgumentParser(description='tpu-perf benchmark tool')
    BuildTree.add_arguments(parser)
    parser.add_argument('--cmodel', action='store_true')
    parser.add_argument('--mlir', action='store_true')
    args = parser.parse_args()
    global option_cmodel_stats
    option_cmodel_stats = args.cmodel

    if not check_buildtree():
        sys.exit(1)

    tree = BuildTree(os.path.abspath('.'), args)
    stat_fn = os.path.join(tree.global_config['outdir'], 'stats.csv')
    extra = set()
    if args.mlir:
        run_func = run_mlir
    else:
        run_func = run_nntc
        for path, config in tree.walk():
            for k in collect_nntc_headers(tree, config):
                extra.add(k)
    extra = list(extra)
    extra.sort()
    with open(stat_fn, 'w') as f:
        csv_f = csv.writer(f)
        if option_cmodel_stats:
            csv_f.writerow([
                'name',
                *extra,
                'shape',
                'gops',
                'time(ms)',
                'cmodel_estimated_time(ms)',
                'mac_utilization',
                'cpu_usage',
                'cmodel_estimated_mac_utilization',
                'ddr_utilization',
                'cmodel_estimated_ddr_bandwidth'])
        else:
            csv_f.writerow([
                'name',
                *extra,
                'shape',
                'gops',
                'time(ms)',
                'mac_utilization',
                'cpu_usage',
                'ddr_utilization'])

        for path, config in tree.walk():
            run_func(tree, path, config, csv_f, extra)

if __name__ == '__main__':
    main()
