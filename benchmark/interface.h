#ifndef _BMRT_BENCHMARK_INTERFACE
#define _BMRT_BENCHMARK_INTERFACE
#ifdef _MSC_VER
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

#include "runner.h"
#ifdef __cplusplus
extern "C"{
#endif

#define MAX_DEVICES_NUM 64
#define MAX_INPUTS 8
#define MAX_STR_LEN 64

typedef struct Latency_And_Throughput{
    float avg_latency;
    float throughput;
    char shape_info[MAX_STR_LEN];
} LatRes;

typedef struct ParamIn{
    int loop_num;
    int enable_copy;
    int use_dev_nums;
    int dev_ids[MAX_DEVICES_NUM];
    const char* bmodel_fn;
} ParamIn;

DLL_EXPORT int getDevNum();
DLL_EXPORT int getPerformance(ParamIn* param, LatRes* res);



#ifdef __cplusplus
}
#endif



#endif