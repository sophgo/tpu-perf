import os
import re
import subprocess
import logging
from pprint import pprint, pformat
from .util import hash

def bulkize(l, n):
    for i in range(0, len(l), n):
        end = min(len(l), i + n)
        yield l[i:end]

def sys_memory_size():
    with open('/proc/meminfo') as f:
       line = next(f)
       m = re.match('^MemTotal:\s+(\d+) kB', line)
       if not m:
           logging.error('Failed to parse memory info')
           raise RuntimeError
       return int(m.group(1))

def env_list_to_dict(env, base=os.environ):
    env_dict = base.copy()
    for v in env:
        pair = v.split('=')
        env_dict[pair[0].strip()] = pair[1].strip() if len(pair) > 1 else ""
    return env_dict

class CommandExecutor:
    def __init__(self, cwd, env=dict(), memory_hint = None, verbose=False, incremental=False):
        self.verbose = verbose
        if memory_hint is None:
            memory_hint = 1024 * 1024 * 7
        self.env = env_list_to_dict(env)

        mem_size = sys_memory_size()
        max_threads = max(1, int(mem_size / memory_hint))
        self.threads = 4
        if self.threads > max_threads:
            self.threads = max_threads
        self.cwd = cwd
        self.incremental = incremental
        self.procs = []
        self.pipes = []
        self.logs = []

    def run(self, *args, **kw_args):
        self.put(*args, **kw_args)
        self.wait()

    def put(self, title, *args, **kw_args):
        if 'cwd' not in kw_args:
            kw_args['cwd'] = self.cwd
        if 'shell' not in kw_args:
            kw_args['shell'] = True
        kw_args['env'] = env_list_to_dict(kw_args['env'], self.env) \
            if kw_args.get('env') \
            else self.env
        self.procs.append((title, args, kw_args))

    def fire(self, bulk = None):
        if bulk is None:
            bulk = self.procs
        self.logs = []
        self.pipes = []
        for title, args, kw_args in bulk:
            cmd_fn = os.path.join(kw_args['cwd'], f'{title}.cmd')
            log_fn = os.path.join(kw_args['cwd'], f'{title}.log')
            flag_fn = f'{cmd_fn}.1'
            
            old_hash = ''
            if os.path.exists(cmd_fn):
                with open(cmd_fn) as r:
                    old_hash = hash(''.join(r.readlines()))
            
            cmd_log = '\n'.join([
                pformat(args),
                pformat(kw_args),
                f'\n\n---------------\n{args}\n',
            ])
            
            cmd_hash = hash(cmd_log)
            if self.incremental and cmd_hash == old_hash and not os.path.exists(flag_fn):
                continue
            
            with open(cmd_fn, 'w') as f:
                f.write(cmd_log)
                
            open(flag_fn,'w').close()
            
            log = open(log_fn, 'w')
            p = subprocess.Popen(*args, **kw_args, stdout=log, stderr=log)
            self.logs.append((log_fn, flag_fn, log))
            self.pipes.append(p)

    def drain(self):
        for i, p in enumerate(self.pipes):
            log_fn, flag_fn, log = self.logs[i]
            ret = p.wait()
            log.close()
            if ret != 0:
                if self.verbose:
                    with open(log_fn) as f:
                        logging.error(f'Command failed-------------->\n{f.read()}')
                else:
                    logging.error(f'Command failed, please check {log_fn}')
                raise RuntimeError('Command failed')
            else:
                os.remove(flag_fn)
                

    def wait(self):
        try:
            for bulk in bulkize(self.procs, self.threads):
                self.fire(bulk)
                self.drain()
        finally:
            self.procs.clear()
