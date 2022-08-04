#!/usr/bin/env python3

import re
import sys
import os
from setuptools import setup, find_packages

import subprocess

def get_version_from_tag():
    ret, val = subprocess.getstatusoutput('git describe --tags')
    if ret != 0:
        return '9.9.9'
    m = re.match('v(\d+\.\d+\.\d+)(-.+)*', val)
    if not m:
        return '9.9.9'
    ver = m.group(1)
    revision = m.group(2)
    if revision:
        revision = revision[1:]
        return f'{ver}.{revision}'
    else:
        return ver

def iter_shared_objects():
    cur_dir = os.path.abspath(os.path.dirname(sys.argv[0]) or '.')
    for dirpath, dirnames, filenames in os.walk(cur_dir):
        for fn in filenames:
            if fn.endswith('.so'):
                yield os.path.join(dirpath, fn)

packages = ['tpu_perf']
so_list = list(iter_shared_objects())

setup(
    version=get_version_from_tag(),
    author='sophgo',
    description='TPU performance benchmark tool',
    author_email='dev@sophgo.com',
    license='Apache',
    name='tpu_perf',
    url='https://www.sophgo.com/',
    install_requires=['numpy', 'lmdb', 'protobuf==3.19.*', 'psutil', 'pyyaml'],
    packages=find_packages(),
    include_package_data=True,
    package_data={'tpu_perf': so_list})
