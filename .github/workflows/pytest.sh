#!/bin/bash

WORKFLOW_DIR="$(dirname "$(realpath "$0")")"
export DIR="$(realpath $WORKFLOW_DIR/../../..)"
export MODEL_ZOO_PATH=$DIR

CI_ENV=$DIR/../../ci.env

[ -f $CI_ENV ] && source $CI_ENV

set -eE

DEBIAN_FRONTEND=noninteractive
[ "$EUID" -eq 0 ] || sudo="sudo -E"
$sudo apt-get install -y libhdf5-dev libatlas-base-dev libboost-system-dev
$sudo apt-get install -y g++-aarch64-linux-gnu gcc-aarch64-linux-gnu
$sudo apt-get install -y libgl1 wget unzip

python3 -m pip install -U pip
python3 -m pip install setuptools wheel
python3 -m pip install -r $WORKFLOW_DIR/../../pytest/requirements.txt

libsophon_dir=/tmp/libsophon

wget="wget -q"

task=$1

build_tpu-perf(){
    # download libsophon
    if [ $LIBSOPHON_URL ]; then
        libsophon_tar=$DIR/libsophon.tar.gz
        $wget -q -O $libsophon_tar $LIBSOPHON_URL
    else
        exit 1
    fi
    mkdir -p $libsophon_dir
    tar --wildcards --strip-components=4 -xf $libsophon_tar \
        -C $libsophon_dir libsophon_*_x86_64/opt/sophon/libsophon*

    # download protoc
    if [ ! "$(protoc --version | head -n1 | cut -d" " -f2)" == "3.19.4" ]; then
        protoc_zip=protoc-3.19.4-linux-x86_64.zip
        $wget -O /tmp/$protoc_zip \
            https://github.com/protocolbuffers/protobuf/releases/download/v3.19.4/$protoc_zip
        unzip -o -d /tmp/ /tmp/$protoc_zip
        PATH=/tmp/bin:$PATH
    fi

    # aarch64
    mkdir -p $DIR/tpu-perf/aarch64_build
    pushd $DIR/tpu-perf/aarch64_build
    cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/aarch64-toolchain.cmake -Dsg_PATH=$libsophon_dir ..
    make install/strip
    popd

    # x86_64
    mkdir -p $DIR/tpu-perf/build
    pushd $DIR/tpu-perf/build
    cmake -Dsg_PATH=$libsophon_dir ..
    make install/strip
    popd

    for fn in $DIR/tpu-perf/python/dist/tpu_perf*.whl; do whl_file=$fn; done
    python3 -m pip uninstall -y $whl_file
    python3 -m pip install --user $whl_file
}

setup_nntc(){
    # download tpu-nntc
    if [ $TPU_NNTC_URL ]; then
        tpu_nntc_tar=$DIR/tpu_nntc.tar.gz
        $wget -O $tpu_nntc_tar $TPU_NNTC_URL > /dev/null
    else
        exit 1
    fi
    tpu_nntc_parent=$DIR/tpu_nntc_sdk
    mkdir -p $tpu_nntc_parent
    tar -xf $tpu_nntc_tar -C $tpu_nntc_parent
    for fn in $tpu_nntc_parent/*; do
        [ ! -d $fn ] && continue
        tpu_nntc_dir=$fn
    done

    export NNTC_TOP=$(realpath $tpu_nntc_dir)
    for w in $tpu_nntc_dir/wheel/ufwio*.whl; do
        python3 -m pip uninstall -y $w;
        python3 -m pip install --user $w
    done
    for w in $tpu_nntc_dir/wheel/*; do
        python3 -m pip uninstall -y $w;
        python3 -m pip install --user $w;
    done
    export LD_LIBRARY_PATH=$(realpath $tpu_nntc_dir)/lib:$libsophon_dir/lib
    export PATH=$(realpath $tpu_nntc_dir)/bin:$PATH
    export PATH=$libsophon_dir/bin:$PATH
}

setup_mlir(){
    # download tpu-mlir
    if [ $TPU_MLIR_URL ]; then
        tpu_mlir_tar=$DIR/tpu_mlir.tar.gz
        $wget -O $tpu_mlir_tar $TPU_MLIR_URL > /dev/null
    else
        exit 1
    fi
    tpu_mlir_parent=$DIR/tpu_mlir_sdk
    mkdir -p $tpu_mlir_parent
    tar -xf $tpu_mlir_tar -C $tpu_mlir_parent
    for fn in $tpu_mlir_parent/*; do
        [ ! -d $fn ] && continue
        tpu_mlir_dir=$fn
    done

    export TPUC_ROOT=$(realpath $tpu_mlir_dir)

    export PATH=$libsophon_dir/bin:$PATH
    export PATH=${TPUC_ROOT}/bin:$PATH
    export PATH=${TPUC_ROOT}/python/tools:$PATH
    export PATH=${TPUC_ROOT}/python/utils:$PATH
    export PATH=${TPUC_ROOT}/python/test:$PATH
    export PATH=${TPUC_ROOT}/python/samples:$PATH
    export LD_LIBRARY_PATH=$TPUC_ROOT/lib:$libsophon_dir/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=${TPUC_ROOT}/python:$PYTHONPATH
}

download_dataset(){
    /bin/bash $DIR/tpu-perf/.github/workflows/download_dataset.sh
}


if [ "$task" == "build" ]; then
    build_tpu-perf
elif [ "$task" == "dataset" ]; then
    download_dataset
elif [ "$task" == "nntc" ]; then
    setup_nntc
    pushd $WORKFLOW_DIR/../../pytest
    python3 -m pytest -m nntc
    popd
elif [ "$task" == "mlir" ]; then
    setup_mlir
    pushd $WORKFLOW_DIR/../../pytest
    python3 -m pytest -m mlir
    popd
fi