import os
import ctypes as ct
import numpy as np
import time
import psutil

## Dtype
class ArgParam(ct.Structure):
    _fields_=[
                ("loop_num", ct.c_int32),
                ("enable_copy", ct.c_int32),
                ("use_dev_nums", ct.c_int32),
                ("dev_ids", ct.c_int32 * 64),
                ("bmodel_fn", ct.c_char_p)
            ]
     
class Res(ct.Structure):
    _fields_ = [
        ("avg_latency", ct.c_float),
        ("throughput", ct.c_float),
        ("shape_info", ct.c_char * 64),
    ]

class BenchmarkInfer:
    __lib = None
    _MAX_NUM_DEVICES = 64
    def __init__(self, bmodel_path=None, batch=1, devices = None) -> None:
        self.bmodel_path = bmodel_path
        if self.__class__.__lib is None:
            lib_path = os.path.join(os.path.dirname(__file__), "libbenchmark.so")
            self.__class__.__lib = ct.cdll.LoadLibrary(lib_path)
    
    @classmethod
    def available_devices(cls):
        if cls.__lib is None:
            lib_path = os.path.join(os.path.dirname(__file__), "libbenchmark.so")
            cls.__lib = ct.cdll.LoadLibrary(lib_path)
        max_num = ct.c_int(cls._MAX_NUM_DEVICES);
        devices = (ct.c_int*max_num.value)()
        real_num = cls.__lib.getDevNum()
        return list(devices[i] for i in range(real_num))

    def getPerformace(self, loop, bmodel, devs = []):
        self.__lib.getPerformance.argtypes = [ct.POINTER(ArgParam), ct.POINTER(Res)]
        param = ArgParam()
        param.loop_num = ct.c_int32(int(loop))
        param.enable_copy = ct.c_int32(1)
        param.use_dev_nums = ct.c_int32(len(devs))
        for i, dev_id in enumerate(devs):
            param.dev_ids[i] = ct.c_int32(dev_id)
        param.bmodel_fn = ct.c_char_p(bmodel.encode())
        res = Res()
        p = psutil.Process(psutil.Process().pid)
        cpu_percent = p.cpu_percent() / 100
        self.__lib.getPerformance(ct.byref(param), ct.byref(res))
        cpu_percent = p.cpu_percent() / 100
        r = {}
        r["avg_latency"] = float(res.avg_latency)
        r["throughput"] = float(res.throughput)
        r["shape"] = res.shape_info.decode('utf-8', errors='replace')
        r["cpu_usage"] = cpu_percent
        return r

if __name__ == "__main__":
    bmodel = "/home/huyu/model-zoo/output/bert_paddle/1b.fp.compilation/compilation.bmodel"
    perf_bench = BenchmarkInfer()
    devs = perf_bench.available_devices()
    print(devs)
    res = perf_bench.getPerformace(100, bmodel, list(devs)) 
    print(res)
