import os
import pytest
from utils import change_dir

@pytest.fixture(scope='session')
def setup_session(pytestconfig, request):
    cwd = os.getcwd()
    change_dir(os.environ.get('MODEL_ZOO_PATH'))
    task = pytestconfig.getoption('-m')
    test_models = ['vision/detection/yolov5']
    test_case_txt = './ci_test_case.txt'
    test_case_txt = os.path.abspath(test_case_txt)
    with open(test_case_txt, 'w', encoding='utf-8') as f:
        for model in test_models:
            f.write(model.strip() + '\n')

    def teardown_session():
        change_dir(cwd)

    request.addfinalizer(teardown_session)

    return task, test_case_txt