import os
import re
import glob
import yaml
import copy
import logging

def read_config(path):
    fn = os.path.join(path, 'config.yaml')
    if not os.path.exists(fn):
        logging.warning(f'No config in {path}')
        return
    with open(fn) as f:
        return yaml.load(f, yaml.Loader)

def hash_name(config):
    from hashlib import md5
    m = md5()
    keys = [k for k in config]
    keys.sort()
    for k in keys:
        m.update(str(config[k]).encode())
    return m.hexdigest()

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
    def __init__(self, root):
        self.root = root
        self.global_config = global_config = read_config(root) or dict()
        global_config['root'] = root
        if 'workdir' not in global_config:
            global_config['workdir'] = os.path.join(root, 'output')

        self.output_names = set()

    def read_global_variable(self, name, config = dict()):
        return self.expand_variables(config, self.global_config[name])

    whole_var_pattern = '^\$\(([a-z0-9_]+)\)$'

    def expand_variables(self, config, string, stack=None, shallow=False):
        if type(string) != str:
            return string
        if stack is None:
            stack = set()
        global_config = self.global_config
        var_match = re.match(self.whole_var_pattern, string)
        if var_match is not None:
            var = var_match.group(1)
            value = config.get(var) or global_config.get(var)
            if not value:
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
            if not value:
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

    def read_dir(self, path):
        for fn in os.listdir(path):
            if not fn.endswith('.yaml'):
                continue
            fn = os.path.join(path, fn)
            if not os.path.isfile(fn):
                continue
            for ret in self._read_dir(fn):
                yield ret

    def _read_dir(self, config_fn):
        path = os.path.dirname(config_fn)
        global_config = self.global_config

        with open(config_fn) as f:
            config = yaml.load(f, yaml.Loader)

        if not config:
            return
        if config.get('ignore'):
            return

        # Pre expand non-string variables. Because variable
        # processing logic might rely on this.
        for k in config:
            if type(config[k]) != str:
                continue
            if not re.match(self.whole_var_pattern, config[k]):
                continue
            try:
                config[k] = self.expand_variables(config, config[k], shallow=True)
            except:
                print(k, config[k])
                raise
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
            workdir = os.path.join(global_config['workdir'], name)
            os.makedirs(workdir, exist_ok=True)

            # Default configuration
            config['home'] = os.path.abspath(path)
            config['workdir'] = workdir
            if 'bmnetu_batch_sizes' not in config:
                config['bmnetu_batch_sizes'] = [1]

            if 'lmdb' in config:
                key = hash_name(config['lmdb'])
                config['lmdb_out'] = os.path.join(
                    self.read_global_variable('data_dir'), key)

            yield path, copy.deepcopy(config)

    def walk(self, path=None):
        if path is None:
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
