import os
import re
import csv
import logging
import subprocess
from prettytable import PrettyTable
from typing import Union, Tuple, List

def change_dir(dir):
    os.chdir(dir)
    logging.info('Current working dir: {}'.format(os.getcwd()))

def log_line(line, test_case_filename=None):
    error_regex = re.compile(r'(?i)\b(error|fail(ed)?|fault)\b')
    no_error_regex = re.compile(r'(?i)\b(no error)\b')
    warning_regex = re.compile(r'(?i)\b(warn(ing)?)\b')
    out_line = "{}{}".format('' if not test_case_filename else test_case_filename + ' :', line)
    if not error_regex.search(line) or no_error_regex.search(line):
        if not warning_regex.search(line):
            logging.info(out_line)
        else:
            logging.warning(out_line)
    else:
        logging.error(out_line)

#执行shell命令
def runcmd(
    cmd:str,
    ret_regex_list:list=None,
    bufsize:int=-1,
    shell=False,
) -> Union[int, Tuple[int, List[List[str]]]]:

    if not shell:
        cmd = cmd.split()
    if ret_regex_list:
        ret = [[] for i in range(len(ret_regex_list))]
    logging.info('run command: {}'.format(cmd))
    logging.info('*********** run command start ***********')
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True, universal_newlines=True, bufsize=bufsize, shell=shell)
    for line in proc.stdout:
        line = line.strip()
        if ret_regex_list:
            for i, regex in enumerate(ret_regex_list):
                if type(regex) == str:
                    groups = re.findall(regex, line)
                else:
                    groups = regex.findall(line)
                if not groups:
                    continue
                if len(groups) > 1:
                    logging.error('regex extraction error')
                    logging.error('regex: {}'.format(regex))
                    assert len(groups) > 1
                ret[i].append(groups[0])
        log_line(line)
    proc.communicate()
    logging.info('*********** run command end ***********')
    if ret_regex_list:
        return proc.returncode, ret
    return proc.returncode

def check_bmodel(path='output', is_mlir=False):
    model_list = os.listdir(path)
    check_pass = True
    for model in model_list:
        model_path = os.path.join(path, model)
        if os.path.isfile(model_path):
            continue
        no_compilation = True
        if not is_mlir:
            for subdir in os.listdir(model_path):
                subdir = os.path.join(model_path, subdir)
                if os.path.isfile(subdir):
                    continue
                if not subdir.endswith('.compilation'):
                    continue
                no_compilation = False
                files = os.listdir(subdir)
                if 'compilation.bmodel' not in files:
                    logging.error('Can\'t find compilation.bmodel in {}'.format(subdir))
                    check_pass = False
            if no_compilation:
                logging.error('Can\'t find *.compilation in {}'.format(model))
                check_pass = False
        else:
            check_model_pass = False
            for subfile in os.listdir(model_path):
                subfile = os.path.join(model_path, subfile)
                if os.path.isdir(subfile):
                    continue
                if subfile.endswith('.bmodel'):
                    check_model_pass = True
            if not check_model_pass:
                logging.warning('Can\'t find *.bmodel in {}'.format(model_path))
                check_pass = False
    return check_pass

def check_stat_csv(csv_file, cfg):
    with open(csv_file, 'r', encoding='utf-8') as f:
        csv_data = csv.DictReader(f)

        for row in csv_data:
            model_name = row['name']
            model_shape = row['shape']
            model_gops = row['gops']
            model_time = row['time']
            model_mac_utilization = row['mac_utilization']
            model_cpu_usage = row['cpu_usage']
            model_ddr_utilization = row['ddr_utilization']

            if model_name not in cfg:
                return False
            model_cfg = cfg[model_name]
            if not re.match(r'\d+(x\d+)+', model_shape):
                return False
            model_gops_list = []
            if 'gops' not in model_cfg:
                model_gops_list.append('N/A')
            else:
                if 'bmnetu_batch_sizes' not in model_cfg:
                    model_cfg['bmnetu_batch_sizes'] = [1]
                for b in model_cfg['bmnetu_batch_sizes']:
                    model_gops_list.append(model_cfg['gops'] * b)
            if model_gops not in model_gops_list:
                return False
            try:
                model_time = eval(model_time)
            except BaseException:
                return False
            for r in [model_mac_utilization, model_cpu_usage, model_ddr_utilization]:
                if not re.match(r'\d{1,3}(\.\d+)?|N/A', r):
                    return False
            return True

def csv2str(csv_file, sep=','):
    is_header = True
    table = None
    with open(csv_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(',')
            if is_header:
                table = PrettyTable(line)
                is_header = False
            else:
                table.add_row(line)
    return table
