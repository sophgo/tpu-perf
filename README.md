TPU Perf
========

### Prerequisites

Install protoc version 3.19 in order to be compatible with ufw.

```bash
mkdir build
cd build
cmake ..
make install
```

### Usage

Refer to [example](./example) for how to create test specs.

Execute from SDK environment.

```bash
python3 -m tpu_perf.build --time # To build efficiency test models
python3 -m tpu_perf.build # To build precision test models
python3 -m tpu_perf.run # To run efficency test
python3 -m tpu_perf.precision_benchmark # To run precision benchmark
```

### config.yaml

#### Preset variables

| Field                 | Type      | Usage                                                                         |
|-----------------------|-----------|-------------------------------------------------------------------------------|
| root                  | Preset    | Repo top directory                                                            |
| home                  | Preset    | Current model directory                                                       |
| workdir               | Preset    | Current model working directory                                               |
| lmdb\_out             | Preset    | Dataset output dir. Only set if lmdb field exists.                            |
| shape                 | Preset    | Current shape list. Only set if shapes field exists.                          |
| shape\_key            | Preset    | Current shape string like "1x3x224x224". Only set if shapes field exists.     |
| shape\_param          | Preset    | Current shape string like "[1,3,224,224]". Only set if shapes field exists.   |

#### Common variables

| Field                 | Type      | Usage                                                                         |
|-----------------------|-----------|-------------------------------------------------------------------------------|
| name                  | Required  | Specify network name, should be unique                                        |
| gops                  | Optional  | Specify network FLOPs                                                         |

#### NNTC

| Field                     | Type      | Usage                                                                                     |
|---------------------------|-----------|-------------------------------------------------------------------------------------------|
| precision                 | Optional  | Boolean type. Indicates whether to build precision model when `tpu_perf.build` is called. |
| fp\_compile\_options    | Optional  | Will use this command to build FP32 bmodel.                                               |
| cali                      | Optional  | Calibration command. `tpu_perf.build` will do calibrating if this variable exists.        |
| time\_only\_cali          | Optional  | Calibration command. `tpu_perf.build --time` will do calibrating if this variable exists. |
| bmnetu\_options           | Optional  | bmnetu compile options. Will do int8 compiling if this variable exists                    |
| bmnetu\_batch\_sizes      | Optional  | Specify int8 output batches                                                               |
| harness                   | Optional  | Harness to call when `tpu_perf.precision_benchmark` is called.                            |
| lmdb                      | Optional  | This object is provided to dataset and harness plugins to do preprocess.                  |

#### MLIR

Variables in top level `config.yaml` are also set for all models.
