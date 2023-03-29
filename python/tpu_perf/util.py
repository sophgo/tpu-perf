import logging
import importlib
import os

def str2list(v):
    files = v.split(",")
    files = [s.strip() for s in files]
    while files.count('') > 0:
        files.remove('')
    return files

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
