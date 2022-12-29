#ifdef __cplusplus
extern "C"{
#endif

/*
typedef enum bm_data_type_e {
  BM_FLOAT32 = 0,
  BM_FLOAT16 = 1,
  BM_INT8 = 2,
  BM_UINT8 = 3,
  BM_INT16 = 4,
  BM_UINT16 = 5,
  BM_INT32 = 6,
  BM_UINT32 = 7
} bm_data_type_t;
*/


#define INVALID_TASK_ID 0
struct tensor_data_t {
    unsigned int dims;
    unsigned int shape[8];
    unsigned int dtype;
    unsigned char* data;
};

unsigned int available_devices(unsigned int* devices, unsigned int maxNum);
void runner_use_devices(const unsigned* device_ids, unsigned num);
unsigned int runner_start_with_batch(const char *bmodel, unsigned int batch);
unsigned int runner_start(const char* bmodel);
void runner_stop(unsigned int runner_id);
int runner_empty(unsigned int runner_id);
int runner_all_stopped(size_t runner_id);
void runner_show_status(unsigned int runner_id);

unsigned int runner_put_input(unsigned runner_id, unsigned int input_num, const tensor_data_t* input_tensors, int need_copy);
tensor_data_t *runner_get_output(unsigned runner_id, unsigned int *task_id, unsigned int *output_num, unsigned int *is_valid);
tensor_data_t *runner_try_to_get_output(unsigned runner_id, unsigned int *task_id, unsigned int *output_num, unsigned int *is_valid);
unsigned int runner_release_output(unsigned int output_num, const tensor_data_t *output_data);

struct blob_info_t {
    const char *name;
    int num_dims;
    int dtype;
    int dims[BM_MAX_DIMS_NUM];
    float scale;
};

blob_info_t *get_input_info(unsigned runner_id, unsigned *num);
blob_info_t *get_output_info(unsigned runner_id, unsigned *num);
void release_input_info(unsigned runner_id, blob_info_t *);
void runner_join(unsigned int runner_id);
unsigned *get_runner_durations(unsigned runner_id, unsigned *num);
void release_unsigned_pointer(unsigned *data);

#ifdef __cplusplus
}
#endif
