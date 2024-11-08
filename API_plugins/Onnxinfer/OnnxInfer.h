#ifndef ONNX_INFER
#define ONNX_INFER

#ifndef OS_LINUX
#include <windows.h>
#endif
#include "onnxruntime/onnxruntime_cxx_api.h"
#include "onnxruntime/onnxruntime_session_options_config_keys.h"
#include <string.h>
#include <iostream>
#include <memx/accl/prepost.h>
#include <thread>

typedef struct{
    std::vector<char* > node_names;
    std::vector<std::vector<int64_t>> node_dims;
    std::vector<ONNXTensorElementDataType> node_types;
    std::vector<size_t> tensor_sizes;
    std::vector<Ort::Value> Tensors;
} onnx_struct;

enum class Mode { Input,
                    Output
};

class OnnxInfer : public PrePost{
    private:
        const char* model_path;

        Ort::Session* session;
        Ort::SessionOptions sessionOptions;
        Ort::Env* env;
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
                                            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        Ort::RunOptions runOpts;
        onnx_struct input_struct;
        onnx_struct output_struct;
        size_t num_input_nodes;
        size_t num_output_nodes;
        bool dynamic_out = false;

        void init_obj(const OrtApi g_ort, onnx_struct& onnx_obj,size_t size, Mode mode);
        std::vector<Ort::Value> inputTensors;
        std::vector<Ort::Value> outputTensors;
        std::thread infer_thread;
    public:
        ~OnnxInfer();
        OnnxInfer(const char* model_path, const std::vector<size_t>& out_sizes);
        void runinference(std::vector<MX::Types::FeatureMap<float>*> input, std::vector<MX::Types::FeatureMap<float>*> output)  override;
        // void runinference(std::vector<MX::Types::FeatureMap<uint8_t>*> input, std::vector<MX::Types::FeatureMap<uint8_t>*> output)  override;
        std::vector<std::vector<int64_t>> get_input_shapes() override;
        std::vector<std::vector<int64_t>> get_output_shapes() override;
        std::vector<size_t> get_output_sizes() override;
        std::vector<size_t> get_input_sizes() override;
        std::vector<std::string> get_output_names() override;
        std::vector<std::string> get_input_names() override;
};

#ifndef OS_LINUX
extern "C" __declspec(dllexport) PrePost * createOnnx(const char* model_path, const std::vector<size_t>&out_sizes);
#else
extern "C" {
    PrePost* createOnnx(const char* model_path, const std::vector<size_t>& out_sizes);
}
#endif

#endif
