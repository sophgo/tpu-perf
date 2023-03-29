#ifndef _BMRT_BENCHMARK
#define _BMRT_BENCHMARK
#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <vector>
#include <map>
#include <numeric>

#include <bmlib_runtime.h>
#include <bmruntime_interface.h>
#include <unistd.h>
#include <cstring>

namespace BmrtBenchmark{

#define CALL_CHECK(fn, ...)   \
    do {    \
        auto ret = fn(__VA_ARGS__); \
        if (ret != BM_SUCCESS) \
        {   \
            std::cerr << #fn << " failed " << ret << std::endl; \
            return -1;  \
        }   \
    } while (0);

#define CALL_CHECK_BOOL(fn, ...)   \
    do {    \
        auto ret = fn(__VA_ARGS__); \
        if (!ret)   \
        {   \
            std::cerr << #fn << " failed " << ret << std::endl; \
            return -1;  \
        }   \
    } while (0);


extern int g_error;

template <typename T>
class Queue
{
private:
    bool drain_ = false;
    int limit_;
    std::queue<T> q_;
    std::mutex mut_;
    std::condition_variable cond_;

public:
    Queue(int limit) : limit_(limit)
    {
    }

    void push(T v)
    {
        std::unique_lock<decltype(mut_)> input_lock(mut_);
        cond_.wait(
            input_lock,
            [&]{
                return q_.size() < limit_;
            });
        q_.push(v);
        cond_.notify_one();
    }

    std::shared_ptr<T> pop()
    {
        auto v = std::make_shared<T>();
        std::unique_lock<decltype(mut_)> input_lock(mut_);
        cond_.wait(
            input_lock,
            [&]{
                return q_.size() || drain_;
            });
        if (q_.empty() && drain_) return nullptr;
        *v = q_.front();
        q_.pop();
        cond_.notify_one();
        return v;
    }

    void join()
    {
        std::unique_lock<decltype(mut_)> input_lock(mut_);
        drain_ = true;
        cond_.notify_all();
    }
};

class Runner{
public:
    Runner(int enable_copy, int loop, bm_handle_t dev_handle,
        void *rt_handle, const char *net_name, bm_tensor_t *inputs,
        int input_num, bm_tensor_t *outputs, int output_num,
        void* in_host_mem, void* out_host_mem);
    ~Runner(){}
    void start();
    void join();
    double get_latency()
    {
        flush_latency_buffer();
        return total_latency_;
    }

private:
    int enable_copy_;
    int loop_;
    bm_handle_t dev_handle_;
    void *rt_handle_;
    const char *net_name_;
    bm_tensor_t *inputs_;
    int input_num_;
    bm_tensor_t *outputs_;
    int output_num_;
    void *input_host_mem_, *output_host_mem_;

    Queue<uint64_t> input_q_, output_q_;
    std::thread infer_t_, load_t_, store_t_;

    double total_latency_ = 0;
    uint64_t latency_buffer_ = 0;
    void flush_latency_buffer()
    {
        total_latency_ += latency_buffer_ / 1000.;
        latency_buffer_ = 0;
    }

    void infer();
    void load();
    void store();


};

}; // 

#endif