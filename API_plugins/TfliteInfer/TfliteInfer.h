#ifndef Tflite_INFER
#define Tflite_INFER

#include <string.h>
#include <memx/accl/prepost.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

class TfliteInfer : public PrePost{
    private:
        const char* model_path_;
        std::unique_ptr<tflite::FlatBufferModel> model;
        tflite::ops::builtin::BuiltinOpResolver resolver;
        std::unique_ptr<tflite::Interpreter> interpreter;
        void record_tensor_details();
        int num_inputs;
        int num_outputs;
        std::vector<std::vector<int64_t>> input_shapes;
        std::vector<std::vector<int64_t>> output_shapes;
        std::vector<size_t> input_sizes;
        std::vector<size_t> output_sizes;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
    public:
        ~TfliteInfer();
        TfliteInfer(const char* model_path, const std::vector<size_t>& out_sizes);
        void runinference(std::vector<MX::Types::FeatureMap<float>*> input, std::vector<MX::Types::FeatureMap<float>*> output)  override;
        std::vector<std::vector<int64_t>> get_input_shapes() override;
        void runinference(std::vector<MX::Types::FeatureMap<uint8_t>*> input, std::vector<MX::Types::FeatureMap<uint8_t>*> output)  override;
        std::vector<std::vector<int64_t>> get_output_shapes() override;
        std::vector<size_t> get_output_sizes() override;
        std::vector<size_t> get_input_sizes() override;
        std::vector<std::string> get_output_names() override;
        std::vector<std::string> get_input_names() override;
};

extern "C" {
    PrePost* createTflite(const char* model_path, const std::vector<size_t>& out_sizes);
}

#endif
