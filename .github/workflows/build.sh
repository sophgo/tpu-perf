#!/bin/bash

DIR="$(dirname "$(realpath "$0")")"
DIR="$(realpath $DIR/../..)"

set -eE

DEBIAN_FRONTEND=noninteractive
[ "$EUID" -eq 0 ] || sudo="sudo -E"
$sudo apt-get update
$sudo apt-get install -y libhdf5-dev libatlas-base-dev libboost-system-dev
$sudo apt-get install -y g++-aarch64-linux-gnu gcc-aarch64-linux-gnu

pip3 install setuptools wheel

libsophon_dst=$DIR/dependencies/libsophon
mkdir -p $libsophon_dst
tar --wildcards --strip-components=4 -xvf $DIR/dependencies/libsophon*.tar.gz \
    -C $libsophon_dst libsophon_*_x86_64/opt/sophon/libsophon*

protoc_zip=protoc-3.19.4-linux-x86_64.zip
wget -O /tmp/$protoc_zip \
    https://github.com/protocolbuffers/protobuf/releases/download/v3.19.4/$protoc_zip
unzip -o -d /tmp/ /tmp/$protoc_zip
PATH=$PATH:/tmp/bin

# aarch64
mkdir -p $DIR/aarch64_build
pushd $DIR/aarch64_build
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/aarch64-toolchain.cmake -Dsg_PATH=$libsophon_dst ..
make install/strip
popd

# x86_64
mkdir -p $DIR/build
pushd $DIR/build
cmake -Dsg_PATH=$libsophon_dst ..
make install/strip

pushd $DIR/dependencies
for fn in tpu-nntc*.tar.gz; do nntc_file=$fn; done
tar xf $nntc_file
for fn in tpu-nntc*; do
    [ ! -d $fn ] && continue
    nntc_dir=$fn
done
export NNTC_TOP=$(realpath $nntc_dir)
for w in $nntc_dir/wheel/ufw-*.whl; do
    ufw_whl=$w
done
for w in $nntc_dir/wheel/ufwio*.whl; do
    ufwio_whl=$w
done
pip3 install --user --force $ufw_whl $ufwio_whl
for w in $nntc_dir/wheel/*bmnetc*.whl; do
    pip3 install --user --force $w
done
for w in $nntc_dir/wheel/*bmnetu*.whl; do
    pip3 install --user --force $w
done
PATH=$PATH:$(realpath $nntc_dir)/bin
popd

for fn in $DIR/python/dist/tpu_perf*.whl; do whl_file=$fn; done
pip3 install $whl_file

pushd $DIR/example
python3 -m tpu_perf.build --time
popd
