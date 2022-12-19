import os
import pytest
import logging
import csv
from pytest import assume

from utils import runcmd, check_bmodel, csv2str

@pytest.fixture(scope='module')
def setup_module(setup_session, request):

    logging.info("setup_module [test_precision] called.")

    retcode = runcmd(cmd='python3 -m tpu_perf.make_lmdb')
    with assume: assert retcode == 0

    task, test_case_txt = setup_session
    ret_regex_list=[r'ERROR', r'Command failed, please check (.*)']
    if task == 'nntc':
        retcode, ret_str = runcmd('python3 -m tpu_perf.build --list {}'.format(test_case_txt), ret_regex_list=ret_regex_list)
    else:
        skip_info = f'current task: {task}, test_precision only run for nntc'
        pytest.skip(skip_info)
    with assume:
        assert retcode == 0
        assert len(ret_str[0]) == 0

    def teardown_module():
        logging.info("teardown_module [test_precision] called.")

    request.addfinalizer(teardown_module)

    return test_case_txt

@pytest.mark.nntc
def test_precision(setup_module):
    test_case_txt = setup_module

    output_dir = 'output'

    # check bmodel
    check_bmodel_pass = check_bmodel(output_dir)
    # with assume: assert check_bmodel_pass == True

    ret_regex_list=[r'ERROR', r'Command failed, please check (.*)']
    retcode, ret_str = runcmd('python3 -m tpu_perf.precision_benchmark --list {}'.format(test_case_txt), ret_regex_list=ret_regex_list)
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

        precision = {}
        with open(csv_file, 'r', encoding='utf-8') as f:
            curr_data = csv.DictReader(f)
            for row in curr_data:
                model_type, model_name = row['name'][::-1].split('-', 1)
                model_type, model_name = model_type[::-1], model_name[::-1]
                if model_name not in precision:
                    precision[model_name] = {}
                precision[model_name][model_type] = {}
                for key in row:
                    if key == 'name':
                        continue
                    precision[model_name][model_type][key] = eval(row[key].replace('%', ''))

        for model in precision:
            model_types = list(precision[model].keys())
            for i in range(len(model_type) - 1):
                for j in range(i+1, len(model_types)):
                    for key in precision[model][model_types[0]]:
                        precision_1 = precision[model][model_types[i]][key]
                        precision_2 = precision[model][model_types[j]][key]
                        difference = abs(precision_1 - precision_2)

                        diff_check_pass = True if difference < 5 else False
                        log_level = 'info' if diff_check_pass else 'error'
                        out_log = 'compare {} with {}:\n{} vs {} ---> difference: {:+.2f}'.format(model+'_'+model_types[i]+'_'+key, model+'_'+model_types[j]+'_'+key, precision_1, precision_2, difference)

                        with assume: assert diff_check_pass
                        getattr(logging, log_level)(out_log)

        os.remove(csv_file)