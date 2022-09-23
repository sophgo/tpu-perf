import os
import re
import glob
import yaml
import copy
import logging

from .util import *

def read_config(path):
    fn = os.path.join(path, 'config.yaml')
    if not os.path.exists(fn):
        logging.warning(f'No config in {path}')
        return
    with open(fn) as f:
        return yaml.load(f, yaml.Loader)

def shape_key_and_param(shape):
    if type(shape) != list:
        logging.error(f'shape should be list')
        raise Exception('invalid shape')
    if type(shape[0]) != list:
        shape = [shape]
    shape_param = ','.join(
        f'[{",".join(str(i) for i in s)}]'
        for s in shape)
    shape_key = '-'.join(
        'x'.join(str(i) for i in s)
        for s in shape)
    return shape_key, shape_param

def check_buildtree():
    ok = os.path.exists('config.yaml')
    if not ok:
        logging.error('config.yaml not found')
    return ok

class BuildTree:
    def __init__(self, root, args = None):
        self.root = root
        self.global_config = global_config = read_config(root) or dict()
        global_config['root'] = root
        if 'outdir' not in global_config:
            global_config['outdir'] = os.path.join(root, 'output')

        self.cases = []
        if not args.full:
            if 'default_cases' in self.global_config:
                self.cases = self.global_config['default_cases']
            if args.list:
                with open(args.list) as f:
                    lines = [l.strip(' \n') for l in f.readlines()]
                    lines = [l for l in lines if l]
                self.cases = lines
            if args.models:
                self.cases = args.models
        global_config['target'] = self.target = args.target
        global_config['devices'] = args.devices

        # Target specific configs
        if self.target in global_config:
            specific = global_config.pop(self.target)
            self.global_config = global_config = dict_override(global_config, specific)

        # Expand all $(home) in global config
        # This allow sub directory rejection
        self.global_config = self.expand_all_variables(
            dict(home=self.root), global_config, shallow=True, no_except=True)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            'models', metavar='MODEL', type=str, nargs='*',
            help='model directories to run')
        parser.add_argument('--full', action='store_true', help='Run all cases')
        parser.add_argument('--list', '-l', type=str, help='Case list')
        parser.add_argument('--devices', '-d',
            type=int, nargs='*', help='Devices',
            default=[0])
        parser.add_argument(
            '--target', '-t', type=str, default='BM1684X',
            choices=['BM1684', 'BM1684X'],
            help='Target chip')

    def read_global_variable(self, name, config = dict(), default=None):
        if default is None and name not in self.global_config:
            logging.error(f'Invalid global config field {name}')
            raise RuntimeError('Invalid Field')
        return self.expand_variables(
            config, self.global_config.get(name, default))

    whole_var_pattern = '^\$\(([a-z0-9_]+)\)$'

    def expand_variables(
        self, config, string,
        stack=None, shallow=False, no_except=False):

        if type(string) != str:
            return string
        if stack is None:
            stack = set()
        global_config = self.global_config
        var_match = re.match(self.whole_var_pattern, string)
        if var_match is not None:
            var = var_match.group(1)
            value = config.get(var) or global_config.get(var)
            if value is None:
                if no_except:
                    return string
                logging.error(f'Invalid variable "{var}" in {string}')
                raise Exception('invalid variable')
            if shallow:
                return value
            return self.expand_variables(config, value, set(var))
        out = string
        prog = re.compile('\$\(([\w_]+)\)', re.MULTILINE)
        for m in re.finditer(prog, string):
            raw = m.group(0)
            var = m.group(1)
            value = config.get(var) or global_config.get(var)
            if value is None:
                if no_except:
                    continue
                logging.error(f'Invalid variable "{var}" in {string}')
                raise Exception('invalid variable')
            if var in stack:
                logging.error(f'Cyclic reference "{var}" in {string}, {stack}')
                raise Exception('invalid variable')
            sub_stack = stack.copy()
            sub_stack.add(var)
            if not shallow:
                value = self.expand_variables(config, value, sub_stack)
            out = out.replace(raw, str(value))
        return out

    def expand_all_variables(self, config, data, **kw_args):
        if type(data) == str:
            return self.expand_variables(config, data, **kw_args)
        elif type(data) == list:
            data = data.copy()
            for i in range(len(data)):
                data[i] = self.expand_all_variables(config, data[i], **kw_args)
        elif type(data) == dict:
            data = data.copy()
            for k, v in data.items():
                data[k] = self.expand_all_variables(config, v, **kw_args)
        return data

    def expand_all_whole_variables(self, config, data, **kw_args):
        if type(data) == str:
            if not re.match(self.whole_var_pattern, data):
                return data
            return self.expand_variables(config, data, **kw_args)
        elif type(data) == list:
            data = data.copy()
            for i in range(len(data)):
                data[i] = self.expand_all_whole_variables(config, data[i], **kw_args)
        elif type(data) == dict:
            data = data.copy()
            for k, v in data.items():
                data[k] = self.expand_all_whole_variables(config, v, **kw_args)
        return data

    def read_dir(self, path):
        path = os.path.abspath(path)
        context = dict()
        p = path
        while True:
            p = os.path.dirname(p)
            if p == self.root:
                break
            if len(p) < 2:
                break
            fn = os.path.join(p, 'config.yaml')
            if not os.path.isfile(fn):
                continue
            context = read_config(p) or dict()
            if self.target in context:
                specific = context.pop(self.target)
                context = dict_override(context, specific)
            context = self.expand_all_variables(
                dict(home=p), context, shallow=True, no_except=True)
            break
        for fn in os.listdir(path):
            if not fn.endswith('config.yaml'):
                continue
            fn = os.path.join(path, fn)
            if not os.path.isfile(fn):
                continue
            for ret in self._read_dir(fn, context):
                yield ret

    def hash_name(self, config):
        from hashlib import md5
        m = md5()
        keys = [k for k in config]
        keys.sort()
        for k in keys:
            v = config[k]
            if type(v) == str and '/' in v and os.path.exists(v):
                # v is a path
                # Convert to relative path
                v = os.path.relpath(v, self.root)
            m.update(f'{k}: {v};'.encode())
        return m.hexdigest()

    def _read_dir(self, config_fn, context = dict()):
        path = os.path.dirname(config_fn)
        global_config = self.global_config

        with open(config_fn) as f:
            config = yaml.load(f, yaml.Loader)

        if not config:
            return
        if config.get('ignore'):
            return

        if self.target in config:
            specific = config.pop(self.target)
            config = dict_override(config, specific)

        config = dict_override(config, context)

        if 'name' not in config:
            logging.error(f'Invalid config {config_fn}')
            raise RuntimeError('Invalid config')

        # Pre expand non-string variables. Because variable
        # processing logic might rely on this.
        config = self.expand_all_whole_variables(config, config, shallow=True)

        if 'harness' in config and type(config['harness']['args']) != list:
            config['harness']['args'] = [config['harness']['args']]

        shapes = config.get('shapes', [None])
        gops_list = config.get('gops', [None])
        if type(gops_list) != list:
            gops_list = [gops_list]
        if len(shapes) != len(gops_list):
            logging.error(
                f'{config["name"]}, length of gops list '
                'should be same with shapes')
            raise RuntimeError('invalid gops')

        for shape, gops in zip(shapes, gops_list):
            name = config['name']
            if gops is not None:
                config['gops'] = gops
            if shape is not None:
                ret = shape_key_and_param(shape)
                name = f'{config["name"]}-{ret[0]}'
                config['shape'] = shape
                config['shape_key'], config['shape_param'] = ret
            if name in self.output_names:
                logging.error(f'Duplicate output name {name}')
                raise RuntimeError('invalid output name')
            self.output_names.add(name)
            workdir = os.path.join(global_config['outdir'], name)
            os.makedirs(workdir, exist_ok=True)

            # Default configuration
            config['home'] = os.path.abspath(path)
            config['workdir'] = workdir
            if 'bmnetu_batch_sizes' not in config:
                config['bmnetu_batch_sizes'] = [1]

            if 'input' in config:
                key = self.hash_name(config['input'])
                data_dir = self.read_global_variable(
                    'data_dir', default='$(root)/data')
                config['lmdb_out'] = os.path.join(data_dir, key)

            yield path, copy.deepcopy(config)

    def walk(self, path=None):
        self.output_names = set()
        if path is None:
            if self.cases:
                for path in self.cases:
                    path = os.path.join(self.root, path)
                    for ret in self.read_dir(path):
                        yield ret
                return
            path = self.root
        if not os.path.isdir(path):
            return
        if os.path.basename(path.strip('/')).startswith('.'):
            return
        has_child = False
        for name in os.listdir(path):
            for ret in self.walk(os.path.join(path, name)):
                has_child = True
                yield ret
        if has_child:
            return
        for ret in self.read_dir(path):
            yield ret
