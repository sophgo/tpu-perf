#include "runner.h"
namespace BmrtBenchmark{
int g_error = 0;
const int queue_len = 3;

Runner::Runner(
    int enable_copy, int loop, bm_handle_t dev_handle, void *rt_handle,
    const char *net_name, bm_tensor_t *inputs, int input_num,
    bm_tensor_t *outputs, int output_num, void* in_host_mem, void* out_host_mem):
                    enable_copy_(enable_copy),
                    loop_(loop),
                    dev_handle_(dev_handle),
                    rt_handle_(rt_handle),
                    net_name_(net_name),
                    inputs_(inputs),
                    input_num_(input_num),
                    outputs_(outputs),
                    output_num_(output_num),
                    input_q_(queue_len), output_q_(queue_len),
                    input_host_mem_(in_host_mem), output_host_mem_(out_host_mem)
{
}

void Runner::start()
{
    load_t_ = std::thread(&Runner::load, this);
    infer_t_ = std::thread(&Runner::infer, this);
    store_t_ = std::thread(&Runner::store, this);
}

void Runner::load()
{
    for (int i = 0; i < loop_; ++i)
    {
        // Load input
        auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; enable_copy_ && j < input_num_; ++j)
        {
            auto ret = bm_memcpy_s2d(dev_handle_, inputs_[j].device_mem, input_host_mem_);
            if (ret != BM_SUCCESS)
            {
                std::cerr << "bm_memcpy_s2d failed" << std::endl;
                goto error;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        input_q_.push(dur.count());
    }
    input_q_.join();
    return;

error:
    g_error = 1;
}

void Runner::infer()
{
    while (true)
    {
        // Pop input
        auto task = input_q_.pop();
        if (!task)
            break;

        // Inference
        auto start = std::chrono::high_resolution_clock::now();
        if (enable_copy_ >= 0)
        {
            auto ok = bmrt_launch_tensor_ex(
                rt_handle_,
                net_name_,
                inputs_,
                input_num_,
                outputs_,
                output_num_,
                true,
                true);
            if (!ok)
            {
                std::cerr << "bmrt_launch_tensor failed!" << std::endl;
                goto error;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Push output
        output_q_.push(*task + dur.count());
    }
    output_q_.join();
    return;

error:
    g_error = 1;
}

void Runner::store()
{
    int index = 0;
    while (true)
    {
        // Pop output
        auto task = output_q_.pop();
        if (!task)
            break;

        // Store output
        auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; enable_copy_ && j < output_num_; ++j)
        {
            auto ret = bm_memcpy_d2s(dev_handle_, output_host_mem_, outputs_[j].device_mem);
            if (ret != BM_SUCCESS)
            {
                std::cerr << "bm_memcpy_d2s failed" << std::endl;
                goto error;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        uint64_t latency = *task + dur.count();
        latency_buffer_ += latency;
        if (latency_buffer_ > 100000000)
            flush_latency_buffer();
    }
    return;

error:
    g_error = 1;
}

void Runner::join()
{
    if (load_t_.joinable())
        load_t_.join();
    if (infer_t_.joinable())
        infer_t_.join();
    if (store_t_.joinable())
        store_t_.join();
}

};