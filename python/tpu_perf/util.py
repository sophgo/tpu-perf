import logging
import importlib
import os

def load_plugins(name):
    for dirpath, dirnames, filenames in os.walk('.'):
        if len(dirpath) >= 2 and dirpath[2] == '.':
            continue
        for dn in dirnames:
            if dn != name:
                continue
            rel_path = os.path.relpath(os.path.join(dirpath, dn), '.')
            import_path = rel_path.replace('/', '.')
            try:
                importlib.import_module(import_path)
            except ModuleNotFoundError as err:
                if err.name not in import_path:
                    raise err
                logging.warning(f'No {name} plugin in {rel_path}')

def dict_override(a, b):
    r = a.copy()
    if type(b) != dict:
        logging.error(f'\"{b}\" is not a dict')
        raise ValueError('Invalid argument')
    for k, v in b.items():
        if k.startswith('+'):
            real_key = k[1:]
            va = a.get(real_key, [])
            if type(va) != list:
                logging.error(f'{real_key} in {a} is not a list')
                raise ValueError('Invalid argument')
            if type(v) != list:
                logging.error(f'{real_key} in {b} is not a list')
                raise ValueError('Invalid argument')
            r[real_key] = va + v
            continue
        r[k] = v
    return r

from datetime import timedelta
def format_seconds(s):
    days = s // (24 * 60 * 60)
    delta = timedelta(seconds=s % (24 * 60 * 60))
    pairs = zip(
        [int(v.split('.')[0]) for v in str(delta).split(':')],
        ['hours', 'minutes', 'seconds'])
    ret = ' '.join(f'{v} {u}' for v, u in pairs if v) or '0 second'
    return f'{days} days {ret}' if days else ret
