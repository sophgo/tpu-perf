#include "interface.h"
#include <cstdlib>
using namespace BmrtBenchmark;

DLL_EXPORT int getDevNum(){
    int dev_count;
    CALL_CHECK(bm_dev_getcount, &dev_count);
    return dev_count;
}

DLL_EXPORT int getPerformance(ParamIn* param, LatRes* res){
    /* ============ param =========== */
    int loop_num = param->loop_num;
    int enable_copy = param->enable_copy;
    int dev_num = param->use_dev_nums;
    int* dev_ids = param->dev_ids;
    const char* bmodel_fn = param->bmodel_fn;
    std::string shape_info = "";

    int dev_count = getDevNum();
    std::vector<int> device_ids;
    for (int i = 0; i < dev_num; ++i)
        device_ids.push_back(dev_ids[i]);

    std::vector<bm_handle_t> dev_handles(device_ids.size());
    std::vector<void *> rt_handles(device_ids.size());
    unsigned prev_chip, cur_chip;
    for (int i = 0; i < device_ids.size(); ++i)
    {
        int dev_id = device_ids[i];
        CALL_CHECK(bm_dev_request, &dev_handles[i], dev_id);
        rt_handles[i] = bmrt_create(dev_handles[i]);

        CALL_CHECK(bm_get_chipid, dev_handles[i], &cur_chip);
        if (i != 0 && prev_chip != cur_chip)
        {
            std::cerr << "Inconsistent chips" << std::endl;
            return -1;
        }
        prev_chip = cur_chip;

        CALL_CHECK_BOOL(bmrt_load_bmodel, rt_handles[i], bmodel_fn);
    }

    const char **network_names;
    bmrt_get_network_names(rt_handles[0], &network_names);
    const bm_net_info_t *net_info = bmrt_get_network_info(
        rt_handles[0], network_names[0]);

    // First dimension stage
    // Second dimension io num
    // Third dimension devices
    std::vector<std::vector<std::vector<bm_tensor_t>>> input_tensors;
    std::vector<std::vector<std::vector<bm_tensor_t>>> output_tensors;

    size_t largest_input = 0, largest_output = 0;
    std::map<int, int> model_batch_to_index;
    const bm_shape_t *input_shapes;
    for (int i_stage = 0; i_stage < net_info->stage_num; ++i_stage)
    {
        size_t in_total = 0, out_total = 0;
        bm_stage_info_t &stage = net_info->stages[i_stage];

        input_tensors.emplace_back();
        auto &stage_in_tensors = input_tensors.back();
        stage_in_tensors.resize(net_info->input_num);
        for (int i_input = 0; i_input < net_info->input_num; ++i_input)
        {
            bm_shape_t shape = stage.input_shapes[i_input];
            for (int i=0; i < shape.num_dims-1; i++){
                shape_info += std::to_string(shape.dims[i]);
                shape_info += "x";
            }
            shape_info += std::to_string(shape.dims[shape.num_dims-1]);
            shape_info += ";";

            if (i_input == 0)
                model_batch_to_index[shape.dims[0]] = i_stage;

            // Multi stage sanity check
            if (i_stage != 0)
            {
                bm_shape_t prev_shape = input_shapes[i_input];
                if (prev_shape.num_dims != shape.num_dims)
                {
                    std::cerr << "Input num dims mismatch" << std::endl;
                    return -1;
                }
                for (int i_dim = 1; i_dim < shape.num_dims; ++i_dim)
                {
                    if (shape.dims[i_dim] != prev_shape.dims[i_dim])
                    {
                        std::cerr << "Input shape mismatch" << std::endl;
                        return -1;
                    }
                }
            }

            bm_data_type_t input_dtype = net_info->input_dtypes[i_input];
            auto &tensors = stage_in_tensors[i_input];
            tensors.resize(device_ids.size());
            for (int i_dev = 0; i_dev < device_ids.size(); ++i_dev)
            {
                auto t = &tensors[i_dev];
                CALL_CHECK_BOOL(
                    bmrt_tensor,
                    &tensors[i_dev],
                    rt_handles[i_dev],
                    input_dtype,
                    shape);
            }
            auto n = bmrt_tensor_bytesize(&tensors[0]);
            if (n > largest_input)
                largest_input = n;
            in_total += n;
        }
        input_shapes = stage.input_shapes;

        output_tensors.emplace_back();
        auto &stage_out_tensors = output_tensors.back();
        stage_out_tensors.resize(net_info->output_num);
        for (int i_output = 0; i_output < net_info->output_num; ++i_output)
        {
            bm_shape_t shape = stage.output_shapes[i_output];
            bm_data_type_t output_dtype = net_info->output_dtypes[i_output];
            auto &tensors = stage_out_tensors[i_output];
            tensors.resize(device_ids.size());
            for (int i_dev = 0; i_dev < device_ids.size(); ++i_dev)
            {
                CALL_CHECK_BOOL(
                    bmrt_tensor,
                    &tensors[i_dev],
                    rt_handles[i_dev],
                    output_dtype,
                    shape);
            }
            auto n = bmrt_tensor_bytesize(&tensors[0]);
            if (n > largest_output)
                largest_output = n;
            out_total += n;
        }
        // std::cout << "In " << in_total << "\t, out " << out_total
        //           << "\tin stage " << i_stage << std::endl;
    }

    char* input_host_mem = new char[largest_input];
    memset(input_host_mem, 0, largest_input);
    char* output_host_mem = new char[largest_output];

    int run_batch = input_shapes[0].dims[0];
    if (model_batch_to_index.find(run_batch) == model_batch_to_index.end())
    {
        std::cerr << "ceil(batch / dev_num) = " << run_batch
                  << " not found in stages" << std::endl;
        return -1;
    }
    int run_stage = model_batch_to_index[run_batch];
    // std::cout << "Run stage " << run_stage << std::endl;
    // std::cout << "Run batch " << run_batch << std::endl;

    std::vector<std::vector<bm_tensor_t>> specific_input_tensors;
    std::vector<std::vector<bm_tensor_t>> specific_output_tensors;
    specific_input_tensors.resize(device_ids.size());
    specific_output_tensors.resize(device_ids.size());
    for (int i_dev = 0; i_dev < device_ids.size(); ++i_dev)
    {
        for (int i = 0; i < net_info->input_num; ++i)
            specific_input_tensors[i_dev].push_back(
                input_tensors[run_stage][i][i_dev]);
        for (int i = 0; i < net_info->output_num; ++i)
            specific_output_tensors[i_dev].push_back(
                output_tensors[run_stage][i][i_dev]);
    }

    std::vector<std::shared_ptr<Runner>> threads;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i_dev = 0; i_dev < device_ids.size(); ++i_dev)
    {
        auto t = std::make_shared<Runner>(
            enable_copy,
            loop_num,
            dev_handles[i_dev],
            rt_handles[i_dev],
            network_names[0],
            specific_input_tensors[i_dev].data(),
            net_info->input_num,
            specific_output_tensors[i_dev].data(),
            net_info->output_num,
            input_host_mem,
            output_host_mem);
        t->start();
        threads.push_back(t);
    }


    double latency;
    for (auto &t : threads)
    {
        t->join();
        latency += t->get_latency();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double ms = dur.count();
    int total = device_ids.size() * loop_num;
    // std::cout << "Average latency: " << (latency / total) << "ms" << std::endl
    //           << "Throughput: " << (total * run_batch * 1000 / ms) << std::endl;

    res->avg_latency = (latency / total);
    res->throughput = total * run_batch * 1000 / ms;
    size_t len_str = shape_info.size() > MAX_STR_LEN ? MAX_STR_LEN : shape_info.size();
    std::strncpy(res->shape_info, shape_info.c_str(), len_str);
    // std::cout << "shape info : " << shape_info << "\n";

    // Free device memories
    for (int i_dev = 0; i_dev < device_ids.size(); ++i_dev)
    {
        for (int i_stage = 0; i_stage < net_info->stage_num; ++i_stage)
        {
            for (auto &dev_tensors : input_tensors[i_stage])
                bm_free_device(dev_handles[i_dev], dev_tensors[i_dev].device_mem);

            for (auto &dev_tensors : output_tensors[i_stage])
                bm_free_device(dev_handles[i_dev], dev_tensors[i_dev].device_mem);
        }
    }

    for (int i = 0; i < device_ids.size(); ++i)
    {
        bmrt_destroy(rt_handles[i]);
        bm_dev_free(dev_handles[i]);
    }

    delete[] input_host_mem;
    delete[] output_host_mem;
    return 0;
}