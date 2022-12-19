import os
import pytest
import logging
from yaml import load as yaml_load
from pytest import assume
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from utils import runcmd, check_bmodel, check_stat_csv, csv2str

# ********************** fixture **********************

@pytest.fixture(scope='module')
def setup_module(setup_session, request):

    logging.info("setup_module [test_efficiency] called.")

    logging.info(os.environ.get('PATH'))
    logging.info(os.environ.get('LD_LIBRARY_PATH'))

    task, test_case_txt = setup_session
    ret_regex_list=[r'ERROR', r'Command failed, please check (.*)']
    if task == 'nntc':
        retcode, ret_str = runcmd(cmd='python3 -m tpu_perf.build --time --list {}'.format(test_case_txt), ret_regex_list=ret_regex_list)
    elif task == 'mlir':
        retcode, ret_str = runcmd(cmd='python3 -m tpu_perf.build --mlir --list {}'.format(test_case_txt), ret_regex_list=ret_regex_list)
    else:
        logging.error(f'Unrecognized task name: {task}')
    with assume:
        assert retcode == 0
        assert len(ret_str[0]) == 0

    def teardown_module():
        logging.info("teardown_module [test_efficiency] called.")

    request.addfinalizer(teardown_module)

    return task, test_case_txt

@pytest.mark.nntc
@pytest.mark.mlir
def test_efficiency(setup_module):
    task, test_case_txt = setup_module

    output_dir = 'output'

    # check bmodel
    check_bmodel_pass = check_bmodel(output_dir)
    # with assume: assert check_bmodel_pass == True

    ret_regex_list=[r'ERROR', r'Command failed, please check (.*)']
    if task == 'nntc':
        retcode, ret_str = runcmd(cmd='python3 -m tpu_perf.run --list {}'.format(test_case_txt), ret_regex_list=ret_regex_list)
    elif task == 'mlir':
        retcode, ret_str = runcmd(cmd='python3 -m tpu_perf.run --mlir --list {}'.format(test_case_txt), ret_regex_list=ret_regex_list)
    else:
        logging.error(f'Unrecognized task name: {task}')
    with assume:
        assert retcode == 0
        assert len(ret_str[0]) == 0

    files = os.listdir(output_dir)
    csv_files = [os.path.join(output_dir, x) for x in files if x.endswith('.csv')]

    for csv_file in csv_files:

        assert os.path.isfile(csv_file)

        logging.info('content of {}'.format(csv_file))
        csv_str = csv2str(csv_file)
        logging.info(csv_str)

        with open(csv_file, 'r', encoding='utf-8') as f:
            csv_lines = f.readlines()
        assert len(csv_lines) > 1

        if os.path.split(csv_file)[-1] == 'stat.csv':
            model_cfg = {}

            def read_model_cfg(root_path):
                global model_cfg
                for f in os.listdir(root_path):
                    f_path = os.path.join(root_path, f)
                    if os.path.isfile(f_path) and f.endswith('.yaml'):
                        new_cfg = yaml_load(f_path, Loader=Loader)
                        if 'name' in new_cfg:
                            model_cfg[new_cfg['name']] = new_cfg
                    elif os.path.isdir(f_path):
                        read_model_cfg(f_path)

            with open(test_case_txt, 'r', encoding='utf-8') as f:
                test_model_dirs = f.readlines()
            for model_dir in test_model_dirs:
                read_model_cfg(model_dir)

            assert len(model_cfg) > 0

            assert check_stat_csv(csv_file, model_cfg) == True

        os.remove(csv_file)